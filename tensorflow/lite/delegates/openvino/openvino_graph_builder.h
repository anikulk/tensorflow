#ifndef TENSORFLOW_LITE_DELEGATES_OPENVINO_GRAPH_BUILDER_H_
#define TENSORFLOW_LITE_DELEGATES_OPENVINO_GRAPH_BUILDER_H_
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset3.hpp>
#include <openvino/opsets/opset8.hpp>
#include <vector>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/openvino/operations/include/add.h"
#include "tensorflow/lite/delegates/openvino/operations/include/conv2d.h"
#include "tensorflow/lite/delegates/openvino/operations/include/concat.h"
#include "tensorflow/lite/delegates/openvino/operations/include/depthwise_conv2d.h"
#include "tensorflow/lite/delegates/openvino/operations/include/dequantize.h"
#include "tensorflow/lite/delegates/openvino/operations/include/hardswish.h"
#include "tensorflow/lite/delegates/openvino/operations/include/logistic.h"
#include "tensorflow/lite/delegates/openvino/operations/include/resize_bilinear.h"
#include "tensorflow/lite/delegates/openvino/operations/include/relu.h"
#include "tensorflow/lite/delegates/openvino/operations/include/relu6.h"
#include "tensorflow/lite/delegates/openvino/operations/openvino_node_manager.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace openvinodelegate {

class OpenVINOGraphBuilder {
public:
  OpenVINOGraphBuilder(std::unique_ptr<NodeManager> node_manager) {
    node_manager_ = std::move(node_manager);
  }

  TfLiteStatus AddInputParams(const TfLiteOpaqueTensor* t, const int index) {
        int32_t num_dims = TfLiteOpaqueTensorNumDims(t);
        std::vector<int> dims(num_dims);
        for (int i = 0; i < num_dims; i++) {
            dims[i]  = TfLiteOpaqueTensorDim(t,i);
	}

        if (dims.size() <= 0) return kTfLiteError;

        auto input = std::make_shared<ov::opset3::Parameter>(ov::element::f32,
                                                             ov::Shape(dims.begin(), dims.end()));
        if (input == NULL) {
            TFLITE_LOG(INFO) << "addInputParams input node is null\n";
            return kTfLiteError;
        }
        input_params_.push_back(input);

        if (dims.size() == 4) {
            ov::AxisVector order = {0, 3, 1, 2};
            const auto order_node = std::make_shared<ov::opset8::Constant>(
                ov::element::i64, ov::Shape{order.size()}, order);
            auto interim = std::make_shared<ov::opset3::Transpose>(input, order_node);
            node_manager_->setOutputAtOperandIndex(index, interim);
            return kTfLiteOk;
        }

        node_manager_->setOutputAtOperandIndex(index, input);
        return kTfLiteOk;
    }

    TfLiteStatus CreateConstNode(const TfLiteOpaqueContext* context, const int index) {
        const TfLiteOpaqueTensor* t = TfLiteOpaqueContextGetOpaqueTensor(context, index);
        int32_t num_dims;
        num_dims = TfLiteOpaqueTensorNumDims(t);
        std::vector<int> dims(num_dims);
        for (int i = 0; i < num_dims; i++) {
            dims[i]  = TfLiteOpaqueTensorDim(t,i);
        }

        if (dims.size() <= 0) return kTfLiteError;

        const void* data = TfLiteOpaqueTensorData(t);
        if (data == NULL) {
            return kTfLiteError;
        }

        auto const_node = std::make_shared<ov::opset8::Constant>(
            ov::element::f32, ov::Shape(dims.begin(), dims.end()), data);
        if (const_node == NULL) {
            TFLITE_LOG(INFO) << "Error in creating const node\n";
            return kTfLiteError;
        }
        node_manager_->setOutputAtOperandIndex(index, const_node);
        return kTfLiteOk;
    }

    void UpdateResultNodes(const TfLiteOpaqueContext* context, std::vector<int> outputs) {
        for (auto o : outputs) {
            auto out_node = node_manager_->getInterimNodeOutput(o);
            auto dims = out_node->get_shape();
            if (dims.size() == 4) {
                ov::AxisVector order;
                order = {0, 2, 3, 1};
                const auto order_node = std::make_shared<ov::opset8::Constant>(
                    ov::element::i64, ov::Shape{order.size()}, order);
                out_node = std::make_shared<ov::opset3::Transpose>(out_node, order_node);
            }
            result_nodes_.push_back(out_node);
        }
    }

    std::vector<std::shared_ptr<ov::Node>> getResultNodes() { return result_nodes_; }

    std::vector<std::shared_ptr<ov::opset3::Parameter>> getInputParams() { return input_params_; }

    size_t getNodeManagerSize() const { return node_manager_->getNodeCount(); }

    TfLiteStatus CreateNodeFromTfLiteOp(int node_id, TfLiteRegistrationExternal* registration,
                                        TfLiteOpaqueNode* node, TfLiteOpaqueContext* context);
    std::shared_ptr<OperationsBase> CreateOpClass(int operationIndex,
                                                    TfLiteRegistrationExternal* registration);

private:
    std::shared_ptr<NodeManager> node_manager_;
    std::vector<std::shared_ptr<ov::opset3::Parameter>> input_params_;
    std::vector<std::shared_ptr<ov::Node>> result_nodes_;
};
}  // namespace openvinodelegate
}  // namespace tflite
#endif  // TENSORFLOW_LITE_DELEGATES_OPENVINO_GRAPH_BUILDER_H_
