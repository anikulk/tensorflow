#include "openvino_graph_builder.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus OpenVINOGraphBuilder::createNodeFromTfLiteOp(int node_id,
                                                          TfLiteRegistrationExternal* registration,
                                                          TfLiteOpaqueNode* node,
                                                          TfLiteOpaqueContext* context) {
    auto operationNode = createOpClass(node_id, registration);
    if (!operationNode) return kTfLiteError;
    operationNode->nodeManager = nodeManager;
    operationNode->SetContext(context);
    const int* inputs_data;
    int inputs_size;
    TfLiteStatus tf_status = TfLiteOpaqueNodeInputs(node, &inputs_data, &inputs_size);
    void* builtin_data = TfLiteOpaqueNodeGetBuiltinData(node);
    operationNode->UpdateNodeInfo((void*)(inputs_data), inputs_size, builtin_data);
    resultNode = operationNode->createNode();
    if (resultNode == nullptr) return kTfLiteError;
    const int* outputs;
    int num_outputs;
    tf_status = TfLiteOpaqueNodeOutputs(node,
                                    &outputs,&num_outputs);
    nodeManager->setOutputAtOperandIndex(outputs[0], resultNode);
    return kTfLiteOk;
}
std::shared_ptr<OperationsBase> OpenVINOGraphBuilder::createOpClass(
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
