#include "openvino_graph_builder.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus OpenVINOGraphBuilder::CreateNodeFromTfLiteOp(int node_id,
                                                          TfLiteRegistration* registration,
                                                          TfLiteNode* node,
                                                          TfLiteContext* context) {
    auto operation_node = CreateOpClass(node_id, registration);
    if (!operation_node) return kTfLiteError;
    operation_node->SetGraphData(context, node_manager_.get());
    operation_node->UpdateNodeInfo(node->inputs->data, node->inputs->size, node->builtin_data);
    std::shared_ptr<ov::Node> result_node = operation_node->CreateNode();
    if (result_node == nullptr) return kTfLiteError;
    node_manager_->setOutputAtOperandIndex(node->outputs->data[0], result_node);
    return kTfLiteOk;
}

std::shared_ptr<OperationsBase> OpenVINOGraphBuilder::CreateOpClass(
    int operationIndex, TfLiteRegistration* registration) {
    switch (registration->builtin_code) {
        case kTfLiteBuiltinAdd: {
            return std::make_shared<Add>(operationIndex);
        }
        case kTfLiteBuiltinConv2d: {
            return std::make_shared<Conv2D>(operationIndex);
        }
        case kTfLiteBuiltinDepthwiseConv2d: {
            return std::make_shared<DepthwiseConv2D>(operationIndex);;
        }
        case kTfLiteBuiltinResizeBilinear: {
            return std::make_shared<ResizeBilinear>(operationIndex);;
        }
        default:
            return nullptr;
    }
}

}  // namespace openvinodelegate
}  // namespace tflite
