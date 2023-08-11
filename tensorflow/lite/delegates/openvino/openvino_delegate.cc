/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/delegates/openvino/openvino_delegate.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

//TODO add openvino specific headers
//#include "xnnpack.h"  // from @XNNPACK
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/delegates/xnnpack/quantization_util.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/utils/sparsity_format_converter.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/tools/optimize/reduced_precision_support.h"

namespace tflite {
namespace openvino {
namespace {

// Forward declaration.
TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate);

class Delegate {
  friend class Subgraph;

 public:
  explicit Delegate(const TfLiteOpenVINODelegateOptions* options) {
    TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                         "Created TensorFlow Lite XNNPACK delegate for CPU.");

    options_ =
        options != nullptr ? *options : TfLiteXNNPackDelegateOptionsDefault();
  }

  TfLiteIntArray* PrepareOpsToDelegate(TfLiteContext* context);
  TfLiteDelegate* tflite_delegate() { return &delegate_; }

  bool support_signed_8bit_quantization() const {
    return (options_.flags & TFLITE_XNNPACK_DELEGATE_FLAG_QS8) != 0;
  }

  bool support_unsigned_8bit_quantization() const {
    return (options_.flags & TFLITE_XNNPACK_DELEGATE_FLAG_QU8) != 0;
  }

  bool support_any_8bit_quantization() const {
    return (options_.flags & (TFLITE_XNNPACK_DELEGATE_FLAG_QU8 |
                              TFLITE_XNNPACK_DELEGATE_FLAG_QS8)) != 0;
  }

  bool force_fp16() const {
    return (options_.flags & TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16) != 0;
  }

 private:
  TfLiteDelegate delegate_ = {
      reinterpret_cast<void*>(this),  // .data_
      DelegatePrepare,                // .Prepare
      nullptr,                        // .CopyFromBufferHandle
      nullptr,                        // .CopyToBufferHandle
      nullptr,                        // .FreeBufferHandle
      kTfLiteDelegateFlagsNone,       // .flags
  };

  // Unpacked data for quasi-static tensors, i.e. tensors produced by
  // dequantizing or unpacking static buffers.
  std::vector<char> static_unpacked_data_;
  // Mapping from a tensor index for a quasi-static tensor to the offset to
  // its unpacked data within static_unpacked_data_.
  std::unordered_map<int, size_t> static_unpacked_data_map_;
  // Set of indices of nodes which unpack static data, e.g. Dequantize
  // operators which convert FP16 static weights to FP32. These nodes are simply
  // ignored in the delegate implementation, because their outputs are
  // pre-unpacked in DelegatePrepare.
  std::unordered_set<int> static_unpack_nodes_;
  // Set of indices of tensors with unpacked static sparse weights.
  std::unordered_set<int> static_sparse_weights_;

  TfLiteOpenVINODelegateOptions options_;
};

class Subgraph {
 public:
  static Subgraph* Create(TfLiteContext* context,
                          const TfLiteDelegateParams* params,
                          const Delegate& delegate) {
    // Convert subgraph inputs and outputs to hash sets for faster lookup.
    const std::unordered_set<int> inputs(
        &params->input_tensors->data[0],
        &params->input_tensors->data[params->input_tensors->size]);
    std::unordered_set<int> outputs;
    for (int o = 0; o < params->output_tensors->size; o++) {
      const int output_tensor_idx = params->output_tensors->data[o];
      // Exclude quasi-static tensors which may have become subgraph outputs
      // after partitioning.
      if (delegate.static_unpacked_data_map_.count(output_tensor_idx) == 0) {
        outputs.insert(output_tensor_idx);
      }
    }
    std::unordered_set<int> externals(outputs);

    TfLiteIntArray* execution_plan;
    if (context->GetExecutionPlan(context, &execution_plan) != kTfLiteOk) {
      return nullptr;
    }


    //TODO: Create Ngraph network creator object

    bool has_sparse_weights = false;
    // Detect which tensors are used as inputs or outputs of any subgraph nodes.
    // -1 denotes tensor not used in the subgraph. These indexes will be
    // filtered out and removed later.
    std::vector<int> tensors(context->tensors_size, -1);
    for (int i = 0; i < params->nodes_to_replace->size; i++) {
      const int node_index = params->nodes_to_replace->data[i];

      TfLiteNode* node = nullptr;
      TfLiteRegistration* registration = nullptr;
      if (context->GetNodeAndRegistration(context, node_index, &node,
                                          &registration) != kTfLiteOk) {
        return nullptr;
      }

      // Detect if any of the node's inputs are sparse weights.
      if (!has_sparse_weights) {
        for (int i = 0; i < node->inputs->size; i++) {
          if (delegate.static_sparse_weights_.count(node->inputs->data[i]) !=
              0) {
            has_sparse_weights = true;
          }
        }
      }

      if (delegate.static_unpack_nodes_.count(node_index) != 0) {
        // The node unpacks static input and can be skipped because its input
        // was pre-unpacked in DelegatePrepare.
        continue;
      }

      switch (registration->builtin_code) {
        case kTfLiteBuiltinMean:
        case kTfLiteBuiltinPad:
        case kTfLiteBuiltinReshape:
        case kTfLiteBuiltinResizeBilinear:
          // Ignore the second input (axes, static padding, or new shape),
          // because it is represented as parameters of the XNNPACK operator
          // rather than extra input.
          {
            const int t = node->inputs->data[0];
            tensors[t] = t;
          }
          break;
        case kTfLiteBuiltinSplit:
          // Ignore the first input (split_dim), as it is represented as
          // parameters of the XNNPACK operator rather than extra input.
          {
            const int t = node->inputs->data[1];
            tensors[t] = t;
            break;
          }
        default:
          // All other operators: process all inputs
          for (int k = 0; k < node->inputs->size; k++) {
            if (registration->builtin_code == kTfLiteBuiltinTransposeConv &&
                k == 0) {
              // Ignore the output size parameter (see above).
              continue;
            }
            const int t = node->inputs->data[k];
            if (t >= 0) {
              tensors[t] = t;
            }
          }
      }
      for (int k = 0; k < node->outputs->size; k++) {
        const int t = node->outputs->data[k];
        if (t >= 0) {
          tensors[t] = t;
        }
      }
    }
    // Filter out and remove -1 (unused) indexes.
    tensors.erase(std::remove_if(tensors.begin(), tensors.end(),
                                 [](int i) { return i < 0; }),
                  tensors.end());
    std::sort(tensors.begin(), tensors.end());

    // REVISIT:  XNNPACK Value IDs for TFLite tensors
    std::vector<uint32_t> ov_tensors(tensors.back() + 1);
    for (int t : tensors) {
      //TODO: add datatype for ov tensors
//      xnn_datatype datatype = xnn_datatype_invalid;

      uint32_t flags = 0;
      const void* data = nullptr;
      if (context->tensors[t].allocation_type == kTfLiteMmapRo) {
        data = context->tensors[t].data.raw_const;
      } else {
        // Check for quasi-static data.
        const auto it = delegate.static_unpacked_data_map_.find(t);
        if (it != delegate.static_unpacked_data_map_.end()) {
          data = delegate.static_unpacked_data_.data() + it->second;
        }
      }

      std::vector<size_t> dims(
          &context->tensors[t].dims->data[0],
          &context->tensors[t].dims->data[NumDimensions(&context->tensors[t])]);
    }

      //TODO: define OV specific tensors

    // Create a set of quasi-static tensors for VisitNode function
    std::unordered_set<int> quasi_static_tensors;
    for (const std::pair<const int, size_t>& entry :
         delegate.static_unpacked_data_map_) {
      quasi_static_tensors.insert(entry.first);
    }

    std::shared_ptr<ov::Model> model;
    // Create ngraph nodes for TFLite delegate nodes
    for (int i = 0; i < params->nodes_to_replace->size; i++) {
      const int node_index = params->nodes_to_replace->data[i];
      if (delegate.static_unpack_nodes_.count(node_index)) {
        // The node unpacks static input and can be skipped because its input
        // was pre-unpacked in DelegatePrepare.
        continue;
      }

      TfLiteNode* node = nullptr;
      TfLiteRegistration* registration = nullptr;
      if (context->GetNodeAndRegistration(context, node_index, &node,
                                          &registration) != kTfLiteOk) {
        return nullptr;
      }

      if (VisitNode(subgraph.get(), delegate, context, registration, node,
                    node_index, quasi_static_tensors,
                    ov_tensors) != kTfLiteOk) {
        return nullptr;
      }
    }

    //TODO REVISIT: Create ngraph 
    if (status != xnn_status_success) {
      TF_LITE_KERNEL_LOG(context, "failed to create XNNPACK runtime");
      return nullptr;
    }

    //TODO: Serialize ngraph

    //TODO REVISIT: replaced runtime_ptr with ngraph graph object
    return new Subgraph(delegate, model, externals);
  }

  TfLiteStatus Prepare(TfLiteContext* context) { return kTfLiteOk; }

  TfLiteStatus Invoke(TfLiteContext* context) {
    //TODO: Create infer request on mNetwork(compiled model)

    //TODO: execute infer() on the infer request

    return kTfLiteOk;
  }

  static TfLiteStatus CheckTensorNonDynamicAllocation(
      TfLiteContext* context, const TfLiteTensor& tensor, int tensor_index,
      int node_index) {
    // TODO: remove checks once dynamic tensors are supported
    if (tensor.allocation_type == kTfLiteDynamic) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context,
          "invalid allocation type in tensor #%d in node #%d: "
          "expected non-dynamic tensor",
          tensor_index, node_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckActivation(int node_index,
                                      TfLiteFusedActivation activation) {
    switch (activation) {
      case kTfLiteActNone:
        return kTfLiteOk;
      case kTfLiteActRelu:
        return kTfLiteOk;
      case kTfLiteActReluN1To1:
      case kTfLiteActRelu6:
        return kTfLiteOk;
	//TODO: Check for this in openvino spec
      case kTfLiteActTanh:
//        return CheckWebNNOpSupport(builder, "tanh");
      case kTfLiteActSignBit:
        TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
            "unsupported fused activation (Sign) in node #%d",
            node_index);
        return kTfLiteError;
      case kTfLiteActSigmoid:
        return kTfLiteOk;
      default:
        TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                        "invalid fused activation (%d) in node #%d",
                        static_cast<int>(activation), node_index);
        return kTfLiteError;
    }
  }
 
  static TfLiteStatus CheckNumInputsAndOutputs(TfLiteContext* context,
                                               TfLiteNode* node,
                                               int expected_num_inputs,
                                               int expected_num_outputs,
                                               int node_index) {
    if (node->inputs->size != expected_num_inputs) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context, "unexpected number of inputs (%d != %d) in node #%d",
          node->inputs->size, expected_num_inputs, node_index);
      return kTfLiteError;
    }
    if (node->outputs->size != expected_num_outputs) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context, "unexpected number of outputs (%d != %d) in node #%d",
          node->outputs->size, expected_num_outputs, node_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckTensorFloat32OrQUInt8Type(const Delegate& delegate,
                                                     TfLiteContext* context,
                                                     const TfLiteTensor& tensor,
                                                     int tensor_index,
                                                     int node_index) {
    switch (tensor.type) {
      case kTfLiteFloat32:
        return kTfLiteOk;
      case kTfLiteInt8:
        if (delegate.support_signed_8bit_quantization()) {
          const auto* quantization_params =
              static_cast<const TfLiteAffineQuantization*>(
                  tensor.quantization.params);
          if (tensor.quantization.type != kTfLiteAffineQuantization ||
              quantization_params->quantized_dimension != 0 ||
              quantization_params->scale == nullptr ||
              quantization_params->scale->size != 1) {
            TF_LITE_MAYBE_KERNEL_LOG(
                context,
                "unsupported quantization type %d in tensor #%d in node #%d",
                tensor.quantization.type, tensor_index, node_index);
            return kTfLiteError;
          }
          return kTfLiteOk;
        }
        break;
      case kTfLiteUInt8:
        if (delegate.support_unsigned_8bit_quantization()) {
          const auto* quantization_params =
              static_cast<const TfLiteAffineQuantization*>(
                  tensor.quantization.params);
          if (tensor.quantization.type != kTfLiteAffineQuantization ||
              quantization_params->quantized_dimension != 0 ||
              quantization_params->scale == nullptr ||
              quantization_params->zero_point == nullptr ||
              quantization_params->scale->size != 1 ||
              quantization_params->zero_point->size != 1) {
            TF_LITE_MAYBE_KERNEL_LOG(
                context,
                "unsupported quantization type %d in tensor #%d in node #%d",
                tensor.quantization.type, tensor_index, node_index);
            return kTfLiteError;
          }
          return kTfLiteOk;
        }
        break;
      default:
        break;
    }
    TF_LITE_MAYBE_KERNEL_LOG(
        context, "unsupported type %s in tensor #%d in node #%d",
        TfLiteTypeGetName(tensor.type), tensor_index, node_index);
    return kTfLiteError;
  }


  static TfLiteStatus VisitNode(
      xnn_subgraph_t subgraph, const Delegate& delegate, TfLiteContext* context,
      TfLiteRegistration* registration, TfLiteNode* node, int node_index,
      const std::unordered_set<int>& quasi_static_tensors,
      const std::vector<uint32_t>& xnnpack_tensors) {
    // TFLite context used for logging purposes. When we create a new node
    // (subgraph is non-null), logging context is the same as context, and error
    // messages are passed to TFLite. When we detect supported operations
    // (subgraph is null), logging context is null, and error messages are
    // supressed.
    TfLiteContext* logging_context = subgraph == nullptr ? nullptr : context;
    switch (registration->builtin_code) {
     case kTfLiteBuiltinAdd: {
        const TfLiteAddParams* add_params =
            static_cast<const TfLiteAddParams*>(node->builtin_data);

        return VisitAddNode(subgraph, delegate, logging_context, node_index,
                            node, context->tensors, add_params,
                            xnnpack_tensors);
      }
      default:
        return kTfLiteError;
    }
  }

  //TODO: check if return type is required
  std::shared_ptr<ov::Node> applyActivation(std::shared_ptr<ov::Node> input, TfLiteFusedActivation activation) {
    switch (activation) {
      case kTfLiteActNone:
        return;
      case kTfLiteActRelu:
        return std::make_shared<ov::opset3::Relu>(input);
      case kTfLiteActReluN1To1:
      case kTfLiteActRelu6:
        return std::make_shared<ov::opset3::Clamp>(input);
	//TODO: Check for this in openvino spec
      case kTfLiteActTanh:
        return std::make_shared<ov::opset3::Tanh>(input)
//        return CheckWebNNOpSupport(builder, "tanh");
      case kTfLiteActSignBit:
        TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
            "unsupported fused activation (Sign) in node #%d",
            node_index);
        return nullptr;
      case kTfLiteActSigmoid:
        return std::make_shared<ov::opset3::Sigmoid>(input);
      default:
        TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                        "invalid fused activation (%d) in node #%d",
                        static_cast<int>(activation), node_index);
        return nullptr;
    }

  }

  static TfLiteStatus VisitAddNode(
      TfLiteContext* logging_context,
      int node_index, TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLiteAddParams* add_params,
      std::vector<uint32_t>& operands,
      const bool detect_supported_op) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 2, 1, node_index));

    const TfLiteTensor& input1_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, input1_tensor,
                                       node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input1_tensor, node->inputs->data[0], node_index));

    const TfLiteTensor& input2_tensor = tensors[node->inputs->data[1]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, input2_tensor,
                                       node->inputs->data[1], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input2_tensor, node->inputs->data[1], node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloat32OrQUInt8Type(delegate, logging_context, output_tensor,
                                       node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, node->outputs->data[0], node_index));

    if (detect_supported_op) {
      if (add_params != nullptr) {
        TF_LITE_ENSURE_STATUS(
            CheckActivation(node_index, add_params->activation));
      }
    } else {
      auto size_input1 = input1_tensor.dims->size;
      auto size_input2 = input2_tensor.dims->size;
      std::vector shape_input1;
      for (int i = 0; i < size_input1; i++)
        shape_input1.insert(input1_tensor.dims->data[i]);
      for (int i = 0; i < size_input2; i++)
        shape_input2.insert(input2_tensor.dims->data[i]);

      //TODO: check for shapes supported
      auto inputNode1 = ;
      auto addNode = std::make_shared<ov::opset8::Add>(inputNode1, inputNode2, ov::op::AutoBroadcastType::NUMPY);
      auto resultNode = applyActivation(addNode, add_params->Activation);
    }

    return TfLiteOk;
  }

 private:
  Subgraph(const Delegate& delegate, const std::shared_ptr<const ov::Model>& model,
           const std::unordered_set<int>& externals)
      : model_(model) {
    for (int t : externals) {
      externals_[t] = nullptr;
    }
  }

  // XNNPACK Runtime (subgraph + workspace) with smart-pointer for lifetime
  // management.
  std::shared_ptr<ov::Model> model_;
  // Mapping from TFLite Tensor IDs (same as XNNPACK Value IDs) for
  // input/output tensors in the delegated subgraph to their data locations.
  std::unordered_map<int, void*> externals_;
  // Memory location to use for 0-size extenal tensors, as TFLite init their
  // data pointer to nullptr, and XNNPACK requires valid data pointers.
  char dummy_data_{0};
};

TfLiteIntArray* Delegate::PrepareOpsToDelegate(TfLiteContext* context) {
  // Clear previous data, in case the delegate is reused without re-creation.
  static_unpacked_data_map_.clear();
  static_unpacked_data_.clear();
  static_unpack_nodes_.clear();
  static_sparse_weights_.clear();

  TfLiteIntArray* execution_plan = nullptr;
  if (context->GetExecutionPlan(context, &execution_plan) != kTfLiteOk) {
    TF_LITE_KERNEL_LOG(context, "Unable to get graph execution plan.");
    return nullptr;
  }

  // Mapping for quasi-static (unpacked from static) tensor index to the node
  // index that produced it.
  std::unordered_map<int, int> quasi_static_tensors_producers;
  // Set of all quasi-static tensors in the execution plan.
  std::unordered_set<int> quasi_static_tensors;
  // Set of quasi-static tensors consumed by the delegated nodes.
  std::unordered_set<int> quasi_static_tensors_to_unpack;

  TfLiteIntArray* nodes_to_delegate =
      TfLiteIntArrayCreate(execution_plan->size);
  nodes_to_delegate->size = 0;
  for (int i = 0; i < execution_plan->size; ++i) {
    const int node_index = execution_plan->data[i];

    // Check if TFLite nodes can be delegated to XNNPACK
    TfLiteNode* node = nullptr;
    TfLiteRegistration* registration = nullptr;
    if (context->GetNodeAndRegistration(context, node_index, &node,
                                        &registration) != kTfLiteOk) {
      TF_LITE_KERNEL_LOG(context,
                         "Unable to get node and registration for node %d.",
                         node_index);
      continue;  // Soft error (skip this node).
    }

    // Prepare to unpack FP16/INT8 tensors.
    if (registration->builtin_code == kTfLiteBuiltinDequantize &&
        node->inputs->size == 1 && node->outputs->size == 1) {
      const TfLiteTensor& input_tensor =
          context->tensors[node->inputs->data[0]];
      const TfLiteTensor& output_tensor =
          context->tensors[node->outputs->data[0]];

      bool is_supported_int8_tensor = input_tensor.type == kTfLiteInt8;
      if (is_supported_int8_tensor) {
        const auto* quant_params = static_cast<const TfLiteAffineQuantization*>(
            input_tensor.quantization.params);
        if (quant_params == nullptr) {
          is_supported_int8_tensor = false;
        }
      }
      if (input_tensor.sparsity == nullptr &&
          (input_tensor.allocation_type == kTfLiteMmapRo ||
           quasi_static_tensors.count(node->inputs->data[0]) != 0) &&
          (input_tensor.type == kTfLiteFloat16 || is_supported_int8_tensor) &&
          output_tensor.type == kTfLiteFloat32) {
        static_unpack_nodes_.insert(node_index);
        quasi_static_tensors_producers[node->outputs->data[0]] = node_index;
        quasi_static_tensors.insert(node->outputs->data[0]);

        if (input_tensor.allocation_type != kTfLiteMmapRo) {
          quasi_static_tensors_to_unpack.insert(node->inputs->data[0]);
        }

        // If dequantized input is sparse, so is its output
        if (static_sparse_weights_.count(node->inputs->data[0]) != 0) {
          static_sparse_weights_.insert(node->outputs->data[0]);
        }

        // Skip this node for now. If output of the node is consumed only by
        // delegated nodes, it will be added to nodes_to_delegate in the end.
        continue;
      }
    }

    // Prepare to unpack sparse tensors.
    // TODO(b/157729695): In the future, we also need to handle the case where a
    // sparse tensor is fed to a TFLite op directly, and no Densify() op is
    // inserted. For now this is not a problem because the Conv() op in tflite
    // can only consume dense tensors.
    if (registration->builtin_code == kTfLiteBuiltinDensify &&
        node->inputs->size == 1 && node->outputs->size == 1) {
      const TfLiteTensor& input_tensor =
          context->tensors[node->inputs->data[0]];
      const TfLiteTensor& output_tensor =
          context->tensors[node->outputs->data[0]];

      if (input_tensor.allocation_type == kTfLiteMmapRo &&
          input_tensor.sparsity != nullptr &&
          (input_tensor.type == kTfLiteFloat16 ||
           input_tensor.type == kTfLiteInt8 ||
           input_tensor.type == kTfLiteFloat32) &&
          output_tensor.type == input_tensor.type) {
        static_unpack_nodes_.insert(node_index);
        quasi_static_tensors_producers[node->outputs->data[0]] = node_index;
        quasi_static_tensors.insert(node->outputs->data[0]);
        static_sparse_weights_.insert(node->outputs->data[0]);

        // Skip this node for now. If output of the node is consumed only by
        // delegated nodes, it will be added to nodes_to_delegate in the end.
        continue;
      }
    }

    if (Subgraph::VisitNode(/*subgraph=*/nullptr, /*delegate=*/*this, context,
                            registration, node, node_index,
                            quasi_static_tensors,
                            std::vector<uint32_t>()) != kTfLiteOk) {
      // If a non-delegated node consumes output of a node that unpacks static
      // data, that node shouldn't be delegated.
      for (int j = 0; j < node->inputs->size; j++) {
        const auto it =
            quasi_static_tensors_producers.find(node->inputs->data[j]);
        if (it != quasi_static_tensors_producers.end()) {
          static_unpack_nodes_.erase(it->second);
        }
      }

      // Non-delegatable node is not an error.
      continue;
    }

    for (int j = 0; j < node->inputs->size; j++) {
      if (quasi_static_tensors.count(node->inputs->data[j]) != 0) {
        quasi_static_tensors_to_unpack.insert(node->inputs->data[j]);
      }
    }

    nodes_to_delegate->data[nodes_to_delegate->size++] = node_index;
  }

  // Sort quasi-static tensors to be unpacked by the node index the produced
  // them. This ensures that in situations where quasi-static tensor is
  // produced from another quasi-static tensor, the tensors are unpacked in
  // the original execution plan order.
  std::vector<int> sorted_quasi_static_tensors_to_unpack(
      quasi_static_tensors_to_unpack.cbegin(),
      quasi_static_tensors_to_unpack.cend());
  std::sort(sorted_quasi_static_tensors_to_unpack.begin(),
            sorted_quasi_static_tensors_to_unpack.end(),
            [&quasi_static_tensors_producers](int t1, int t2) {
              return quasi_static_tensors_producers[t1] <
                     quasi_static_tensors_producers[t2];
            });

  // Unpack static data of all tensors
  for (int t : sorted_quasi_static_tensors_to_unpack) {
    const int producer_index = quasi_static_tensors_producers[t];
    // Check if TFLite nodes can be delegated to XNNPACK
    TfLiteNode* node = nullptr;
    TfLiteRegistration* registration = nullptr;
    if (context->GetNodeAndRegistration(context, producer_index, &node,
                                        &registration) != kTfLiteOk) {
      TF_LITE_KERNEL_LOG(context,
                         "Unable to get node and registration for node %d.",
                         producer_index);
      TfLiteIntArrayFree(nodes_to_delegate);
      return nullptr;  // Hard error.
    }

    if (node->inputs->size != 1) {
      TF_LITE_KERNEL_LOG(context, "unexpected number of inputs (%d) in node %d",
                         node->inputs->size, producer_index);
      TfLiteIntArrayFree(nodes_to_delegate);
      return nullptr;  // Hard error.
    }

    if (node->outputs->size != 1) {
      TF_LITE_KERNEL_LOG(context,
                         "unexpected number of outputs (%d) in node %d",
                         node->outputs->size, producer_index);
      TfLiteIntArrayFree(nodes_to_delegate);
      return nullptr;  // Hard error.
    }

    const TfLiteTensor& input_tensor = context->tensors[node->inputs->data[0]];

    // Consider the case when the input to unpacking node is quasi-static.
    const auto static_unpacked_input_it_ =
        static_unpacked_data_map_.find(node->inputs->data[0]);
    if (static_unpacked_input_it_ == static_unpacked_data_map_.end()) {
      if (input_tensor.allocation_type != kTfLiteMmapRo) {
        TF_LITE_KERNEL_LOG(
            context,
            "unexpected allocation type (%d) in tensor %d in node %d (%d)",
            input_tensor.allocation_type, node->inputs->data[0], producer_index,
            registration->builtin_code);
        TfLiteIntArrayFree(nodes_to_delegate);
        return nullptr;  // Hard error.
      }
    }

    const TfLiteTensor& output_tensor = context->tensors[t];
    size_t tensor_elements = output_tensor.bytes;
    switch (output_tensor.type) {
      case kTfLiteFloat32:
        tensor_elements /= sizeof(float);
        break;
      case kTfLiteFloat16:
        tensor_elements /= sizeof(uint16_t);
        break;
      case kTfLiteInt8:
        tensor_elements /= sizeof(int8_t);
        break;
      default: {
        TF_LITE_KERNEL_LOG(context,
                           "unexpected datatype (%s) in tensor %d in node %d",
                           TfLiteTypeGetName(output_tensor.type),
                           node->outputs->data[0], producer_index);
        TfLiteIntArrayFree(nodes_to_delegate);
        return nullptr;  // Hard error.
      }
    }

    // Align to XNN_EXTRA_BYTES bytes
    while (static_unpacked_data_.size() % XNN_EXTRA_BYTES != 0) {
      static_unpacked_data_.push_back(0);
    }
    const size_t tensor_offset = static_unpacked_data_.size();
    static_unpacked_data_.resize(tensor_offset + context->tensors[t].bytes);

    char* unpacked_data = static_unpacked_data_.data() + tensor_offset;
    const char* packed_data =
        static_unpacked_input_it_ != static_unpacked_data_map_.end()
            ? static_unpacked_data_.data() + static_unpacked_input_it_->second
            : static_cast<const char*>(input_tensor.data.data);
    switch (registration->builtin_code) {
      case kTfLiteBuiltinDequantize: {
        // Such a condition has been checked when preparing to unpack FP16/INT8
        // tensors.
        TFLITE_DCHECK(input_tensor.sparsity == nullptr);
        // Actual data unpacking
        switch (input_tensor.type) {
          case kTfLiteFloat16:
            DequantizeFloat16(reinterpret_cast<const uint16_t*>(packed_data),
                              reinterpret_cast<float*>(unpacked_data),
                              tensor_elements);
            break;
          case kTfLiteInt8: {
            TfLiteAffineQuantization* quant_params =
                static_cast<TfLiteAffineQuantization*>(
                    input_tensor.quantization.params);
            // Such conditions have been checked when preparing to unpack INT8
            // tensors.
            TFLITE_DCHECK(quant_params != nullptr);

            if (quant_params->scale->size == 1) {
              // Per-tensor quantization
              DequantizeInt8(reinterpret_cast<const int8_t*>(packed_data),
                             reinterpret_cast<float*>(unpacked_data),
                             GetTensorShape(&input_tensor),
                             input_tensor.params.zero_point,
                             input_tensor.params.scale);
            } else {
              // Per-channel quantization
              PerChannelDequantizeInt8(
                  reinterpret_cast<const int8_t*>(packed_data),
                  reinterpret_cast<float*>(unpacked_data),
                  GetTensorShape(&input_tensor), quant_params->zero_point->data,
                  quant_params->scale->data, quant_params->quantized_dimension);
            }
            break;
          }
          default:
            // This should not happen as we only allow FP16/INT8 input_tensor
            // when preparing the unpacking.
            TFLITE_DCHECK(false);
        }
        break;
      }
      case kTfLiteBuiltinDensify: {
        // Such a condition has been checked when preparing to unpack FP16/INT8
        // tensors.
        TFLITE_DCHECK(input_tensor.sparsity != nullptr);
        const int dims_count = NumDimensions(&output_tensor);
        std::vector<int> vector_shape(dims_count);
        for (int i = 0; i < dims_count; i++) {
          vector_shape[i] = SizeOfDimension(&output_tensor, i);
        }

        switch (input_tensor.type) {
          case kTfLiteFloat32: {
            const size_t dense_size = context->tensors[t].bytes / sizeof(float);
            float* unpacked_fp32_data = reinterpret_cast<float*>(unpacked_data);
            tflite::internal::sparsity::FormatConverter<float> converter(
                vector_shape, *input_tensor.sparsity);
            converter.SparseToDense(
                static_cast<const float*>(input_tensor.data.data), dense_size,
                unpacked_fp32_data, context);
            break;
          }
          case kTfLiteFloat16: {
            const size_t dense_size =
                context->tensors[t].bytes / sizeof(Eigen::half);
            Eigen::half* unpacked_fp16_data =
                reinterpret_cast<Eigen::half*>(unpacked_data);
            tflite::internal::sparsity::FormatConverter<Eigen::half> converter(
                vector_shape, *input_tensor.sparsity);
            converter.SparseToDense(
                static_cast<const Eigen::half*>(input_tensor.data.data),
                dense_size, unpacked_fp16_data, context);
            break;
          }
          case kTfLiteInt8: {
            const size_t dense_size =
                context->tensors[t].bytes / sizeof(int8_t);
            int8_t* unpacked_int8_data =
                reinterpret_cast<int8_t*>(unpacked_data);
            tflite::internal::sparsity::FormatConverter<int8_t> converter(
                vector_shape, *input_tensor.sparsity);
            converter.SparseToDense(
                static_cast<const int8_t*>(input_tensor.data.data), dense_size,
                unpacked_int8_data, context);
            break;
          }
          default: {
            // This should not happen as we only allow FP16/INT8 input_tensor
            // when preparing the unpacking.
            TFLITE_DCHECK(false);
          }
        }
        break;
      }
      default:
        TF_LITE_KERNEL_LOG(context, "unexpected op registration %d at node %d",
                           registration->builtin_code, producer_index);
        TfLiteIntArrayFree(nodes_to_delegate);
        return nullptr;  // Hard error.
    }

    static_unpacked_data_map_[t] = tensor_offset;
  }

  // Add nodes that unpack static data consumed by delegated nodes.
  // Note: this is done purely to avoid the overhead of running these nodes
  // again in TFLite interpreter which would allocate memory for their outputs.
  // We mark them as delegated, but the delegate would simply ignore these nodes
  // as the static weights are already unpacked.
  for (int node_index : static_unpack_nodes_) {
    nodes_to_delegate->data[nodes_to_delegate->size++] = node_index;
  }
  std::sort(&nodes_to_delegate->data[0],
            &nodes_to_delegate->data[nodes_to_delegate->size]);

#ifdef XNNPACK_DELEGATE_TEST_MODE
  // In the test mode build (used by unit tests), XNNPACK delegate claims to
  // support all operators in the execution plan to disable fallback to the
  // default TensorFlow Lite kernels. Thus, if any of the ops in the model are
  // not supported by the delegate, they will cause a failure in
  // ::tflite::Interpreter::ModifyGraphWithDelegate, to be caught in the unit
  // tests.
  nodes_to_delegate->size = execution_plan->size;
  std::copy(&execution_plan->data[0],
            &execution_plan->data[execution_plan->size],
            &nodes_to_delegate->data[0]);
#endif

  return nodes_to_delegate;
}

void* SubgraphInit(TfLiteContext* context, const char* buffer, size_t length) {
  const TfLiteDelegateParams* params =
      reinterpret_cast<const TfLiteDelegateParams*>(buffer);

  return static_cast<void*>(Subgraph::Create(
      context, params,
      *static_cast<::tflite::xnnpack::Delegate*>(params->delegate->data_)));
}

TfLiteStatus SubgraphPrepare(TfLiteContext* context, TfLiteNode* node) {
  if (node->user_data == nullptr) {
    return kTfLiteError;
  }

  return static_cast<Subgraph*>(node->user_data)->Prepare(context);
}

TfLiteStatus SubgraphInvoke(TfLiteContext* context, TfLiteNode* node) {
  if (node->user_data == nullptr) {
    return kTfLiteError;
  }

  return static_cast<Subgraph*>(node->user_data)->Invoke(context);
}

void SubgraphFree(TfLiteContext* context, void* buffer) {
  if (buffer != nullptr) {
    delete static_cast<Subgraph*>(buffer);
  }
}

const TfLiteRegistration kSubgraphRegistration = {
    /*.init=*/SubgraphInit,
    /*.free=*/SubgraphFree,
    /*.prepare=*/SubgraphPrepare,
    /*.invoke=*/SubgraphInvoke,
    /*.profiling_string=*/nullptr,
    /*.builtin_code=*/0,
    /*.custom_name=*/"TfLiteXNNPackDelegate",
    /*.version=*/2,
};

TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
  TfLiteIntArray* ops_to_replace =
      static_cast<::tflite::xnnpack::Delegate*>(delegate->data_)
          ->PrepareOpsToDelegate(context);
  if (ops_to_replace == nullptr) {
    return kTfLiteError;
  }

  const TfLiteStatus status = context->ReplaceNodeSubsetsWithDelegateKernels(
      context, kSubgraphRegistration, ops_to_replace, delegate);
  TfLiteIntArrayFree(ops_to_replace);
  return status;
}

}  // namespace
}  // namespace xnnpack
}  // namespace tflite

TfLiteXNNPackDelegateWeightsCache* TfLiteXNNPackDelegateWeightsCacheCreate() {
  xnn_status status = xnn_initialize(/*allocator=*/nullptr);
  if (status != xnn_status_success) {
    return nullptr;
  }

  xnn_weights_cache_t weights_cache;
  xnn_create_weights_cache(&weights_cache);
  return reinterpret_cast<TfLiteXNNPackDelegateWeightsCache*>(weights_cache);
}

void TfLiteXNNPackWeightsCacheDelete(TfLiteXNNPackDelegateWeightsCache* cache) {
  if (cache == nullptr) {
    return;
  }
  auto weights_cache = reinterpret_cast<xnn_weights_cache_t>(cache);
  xnn_delete_weights_cache(weights_cache);
  xnn_deinitialize();
}

TfLiteXNNPackDelegateOptions TfLiteXNNPackDelegateOptionsDefault() {
  TfLiteXNNPackDelegateOptions options = {0};

  // Quantized inference is enabled by default on Web platform
#ifdef XNNPACK_DELEGATE_ENABLE_QS8
  options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_QS8;
#endif
#ifdef XNNPACK_DELEGATE_ENABLE_QU8
  options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_QU8;
#endif

  // Enable quantized inference for the delegate build used in unit tests.
#ifdef XNNPACK_DELEGATE_TEST_MODE
  options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_QS8;
  options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_QU8;
#endif  // XNNPACK_DELEGATE_TEST_MODE

  return options;
}

TfLiteDelegate* TfLiteXNNPackDelegateCreate(
    const TfLiteXNNPackDelegateOptions* options) {
  xnn_status status = xnn_initialize(/*allocator=*/nullptr);
  if (status != xnn_status_success) {
    return nullptr;
  }

  auto* xnnpack_delegate = new ::tflite::xnnpack::Delegate(options);
  return xnnpack_delegate ? xnnpack_delegate->tflite_delegate() : nullptr;
}

void* TfLiteXNNPackDelegateGetThreadPool(TfLiteDelegate* delegate) {
  if (delegate == nullptr) {
    return nullptr;
  }

  return static_cast<void*>(
      static_cast<::tflite::xnnpack::Delegate*>(delegate->data_)->threadpool());
}

void TfLiteXNNPackDelegateDelete(TfLiteDelegate* delegate) {
  if (delegate != nullptr) {
    delete static_cast<::tflite::xnnpack::Delegate*>(delegate->data_);
  }
}
