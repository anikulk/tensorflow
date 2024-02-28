/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "openvino_delegate.h"

#include "openvino/runtime/core.hpp"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/utils/simple_opaque_delegate.h"

namespace tflite {
namespace openvinodelegate {
TfLiteStatus OpenVINODelegate::CheckInputsType(const TfLiteOpaqueTensor* opaque_tensor,
                                       TfLiteType expected_type, bool& is_supported) const {
    
    if (opaque_tensor == nullptr) return kTfLiteDelegateError;
    TfLiteType type = TfLiteOpaqueTensorType(opaque_tensor);
    is_supported = (expected_type == type);
    return kTfLiteOk;
}

TfLiteStatus OpenVINODelegate::CheckDataTypeSupported(
    const TfLiteOpaqueContext* context, const TfLiteOpaqueNode* node,
    std::vector<std::vector<TfLiteType>> supported_types, bool& is_supported) const {
    const int* inputs;
    int num_inputs;
    auto tf_status = TfLiteOpaqueNodeInputs(node, &inputs, &num_inputs);
    if (tf_status != kTfLiteOk) return tf_status;
    for (int i = 0; i < supported_types.size(); i++) {
        int tensor_id = inputs[i];
        is_supported = false;
        for (TfLiteType type : supported_types[i]) {
            const TfLiteOpaqueTensor* opaque_tensor =
                TfLiteOpaqueContextGetOpaqueTensor(context, tensor_id);
            if (CheckInputsType(opaque_tensor, type, is_supported) != kTfLiteOk)
                return kTfLiteDelegateError;
            if (is_supported == true)
                break;
        }
        if (is_supported == false) return kTfLiteOk;
    }
    return kTfLiteOk;
}

TfLiteStatus OpenVINODelegate::CheckDims(const TfLiteOpaqueContext* context, const TfLiteOpaqueNode* node,
                                 std::vector<std::vector<int>> dims_size, bool& is_supported) const {
    const int* inputs;
    int num_inputs;
    bool supported;
    auto tf_status = TfLiteOpaqueNodeInputs(node, &inputs, &num_inputs);
    for (int i = 0; i < dims_size.size(); i++) {
        is_supported = false;
        const TfLiteOpaqueTensor* opaque_tensor =
            TfLiteOpaqueContextGetOpaqueTensor(context, inputs[i]);
        for (int j = 0; j < dims_size[i].size(); j++) {
            if (TfLiteOpaqueTensorNumDims(opaque_tensor) == dims_size[i][j]) {
                is_supported = true;
                int size = 1;
                for (int k = 0; k < dims_size[i][j]; k++)
                    size *= TfLiteOpaqueTensorDim(opaque_tensor, k);
                if (size == 0) return kTfLiteOk;
            }
        }
        if (is_supported == false) return kTfLiteOk;
    }
    return kTfLiteOk;
}

bool OpenVINODelegate::IsNodeSupportedByDelegate(const TfLiteRegistrationExternal* registration,
                                                 const TfLiteOpaqueNode* node,
                                                 TfLiteOpaqueContext* context) const {
    bool is_supported = false;
    TfLiteStatus tf_status = kTfLiteOk;
    switch (TfLiteRegistrationExternalGetBuiltInCode(registration)) {
        case kTfLiteBuiltinAdd: {
            tf_status = CheckDataTypeSupported(context, node, {{kTfLiteFloat32}, {kTfLiteFloat32}}, is_supported);
        }
        case kTfLiteBuiltinAveragePool2d: {
            tf_status = CheckDataTypeSupported(context, node, {{kTfLiteFloat32}}, is_supported);
        }
        case kTfLiteBuiltinConv2d: {
            tf_status = CheckDataTypeSupported(context, node,
                                          {{kTfLiteFloat32}, {kTfLiteFloat32}, {kTfLiteFloat32}}, is_supported);
            if (tf_status != kTfLiteOk || is_supported == false) {
                return false;
            }
            tf_status =  CheckDims(context, node, {{4}, {4}}, is_supported);
        }
        case kTfLiteBuiltinConcatenation: {
            tf_status = CheckDataTypeSupported(context, node, {{kTfLiteFloat32}, {kTfLiteFloat32}}, is_supported);
        }
        case kTfLiteBuiltinDepthwiseConv2d: {
            tf_status = CheckDataTypeSupported(context, node, {{kTfLiteFloat32}, {kTfLiteFloat32}}, is_supported);
            if (tf_status != kTfLiteOk || is_supported == false) {
                return false;
            }
            tf_status = CheckDims(context, node, {{4}, {4}}, is_supported);
        }
        case kTfLiteBuiltinDequantize: {
            tf_status = CheckDataTypeSupported(context, node, {{kTfLiteFloat16}}, is_supported);
        }
        case kTfLiteBuiltinResizeBilinear: {
            tf_status = CheckDataTypeSupported(context, node, {{kTfLiteFloat32}}, is_supported);
        }
        case kTfLiteBuiltinRelu: {
            tf_status = CheckDataTypeSupported(context, node, {{kTfLiteFloat32}}, is_supported);
        }
        case kTfLiteBuiltinRelu6: {
            tf_status = CheckDataTypeSupported(context, node, {{kTfLiteFloat32}}, is_supported);
        }
        case kTfLiteBuiltinLogistic: {
            tf_status = CheckDataTypeSupported(context, node, {{kTfLiteFloat32}}, is_supported);
        }
        case kTfLiteBuiltinHardSwish: {
            tf_status = CheckDataTypeSupported(context, node, {{kTfLiteFloat32}}, is_supported);
        }
        case kTfLiteBuiltinMul: {
            tf_status = CheckDataTypeSupported(context, node, {{kTfLiteFloat32}, {kTfLiteFloat32}}, is_supported);
        }
        case kTfLiteBuiltinSoftmax: {
            tf_status = CheckDataTypeSupported(context, node, {{kTfLiteFloat32}}, is_supported);
        }
        case kTfLiteBuiltinTanh: {
           tf_status = CheckDataTypeSupported(context, node, {{kTfLiteFloat32}}, is_supported);
        }
        case kTfLiteBuiltinReshape: {
            tf_status = CheckDataTypeSupported(context, node, {{kTfLiteFloat32}, {kTfLiteInt32}}, is_supported);
        }
        case kTfLiteBuiltinMaxPool2d: {
           tf_status = CheckDataTypeSupported(context, node, {{kTfLiteFloat32}}, is_supported);
        }
        case kTfLiteBuiltinMean: {
            tf_status = CheckDataTypeSupported(context, node, {{kTfLiteFloat32}}, is_supported);
            if (tf_status != kTfLiteOk || is_supported == false) {
                return false;
            }
            tf_status = CheckDims(context, node, {{4}, {1}}, is_supported);
        }
        default:
            return false;
    }
    if (tf_status != kTfLiteOk) return false;
    return is_supported;
}

TfLiteStatus OpenVINODelegate::Initialize(TfLiteOpaqueContext* context) { return kTfLiteOk; }

const char* OpenVINODelegate::Name() const { return "OpenVINO SimpleOpaqueDelegate"; }

std::unique_ptr<tflite::SimpleOpaqueDelegateKernelInterface>
OpenVINODelegate::CreateDelegateKernelInterface() {
    return std::unique_ptr<tflite::openvinodelegate::OpenVINODelegateKernel>(
        new tflite::openvinodelegate::OpenVINODelegateKernel());
}
}  // namespace openvinodelegate
}  // namespace tflite

TfLiteDelegate* TFL_CAPI_EXPORT
TfLiteCreateOpenVINODelegate(const TfLiteOpenVINODelegateOptions* options) {
    auto ovdelegate_ = std::make_unique<tflite::openvinodelegate::OpenVINODelegate>(options);
    return tflite::TfLiteOpaqueDelegateFactory::CreateSimpleDelegate(std::move(ovdelegate_));
}

void TFL_CAPI_EXPORT TfLiteDeleteOpenVINODelegate(TfLiteOpaqueDelegate* delegate) { return; }

TfLiteOpenVINODelegateOptions TFL_CAPI_EXPORT TfLiteOpenVINODelegateOptionsDefault() {
    TfLiteOpenVINODelegateOptions result;
    result.debug_level = 0;
    result.plugins_path = "/tmp/plugins.xml";
    return result;
}
