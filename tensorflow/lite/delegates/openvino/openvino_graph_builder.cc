#include "openvino_graph_builder.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus OpenVINOGraphBuilder::CreateNodeFromTfLiteOp(int node_id,
                                                          TfLiteRegistrationExternal* registration,
                                                          TfLiteOpaqueNode* node,
                                                          TfLiteOpaqueContext* context) {
    auto operation_node = CreateOpClass(node_id, registration);
    if (!operation_node) return kTfLiteError;
    operation_node->SetGraphData(context, node_manager_.get());
    const int* inputs_data;
    int num_inputs;
    TfLiteStatus status = TfLiteOpaqueNodeInputs(node, &inputs_data, &num_inputs);
    operation_node->UpdateNodeInfo((void*)inputs_data, num_inputs, TfLiteOpaqueNodeGetBuiltinData(node));
    std::shared_ptr<ov::Node> result_node = operation_node->CreateNode();
    if (result_node == nullptr) return kTfLiteError;
    const int* outputs;
    int num_outputs;
    TfLiteStatus tf_status = TfLiteOpaqueNodeOutputs(node,
                                    &outputs,&num_outputs);
    node_manager_->setOutputAtOperandIndex(outputs[0], result_node);
    return kTfLiteOk;
}

std::shared_ptr<OperationsBase> OpenVINOGraphBuilder::CreateOpClass(
    int operationIndex, TfLiteRegistrationExternal* registration) {
    switch (TfLiteRegistrationExternalGetBuiltInCode(registration)) {
        case kTfLiteBuiltinAdd: {
            return std::make_shared<Add>(operationIndex);
        }
        case kTfLiteBuiltinConv2d: {
            return std::make_shared<Conv2D>(operationIndex);
        }
        case kTfLiteBuiltinDepthwiseConv2d: {
            return std::make_shared<DepthwiseConv2D>(operationIndex);
            ;
        }
        case kTfLiteBuiltinResizeBilinear: {
            return std::make_shared<ResizeBilinear>(operationIndex);
            ;
        }
        default:
            return nullptr;
    }
}

}  // namespace openvinodelegate
}  // namespace tflite
