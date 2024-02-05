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
    operation_node->UpdateNodeInfo((void*)inputs_data, num_inputs,
                                   TfLiteOpaqueNodeGetBuiltinData(node));
    std::shared_ptr<ov::Node> result_node = operation_node->CreateNode();
    if (result_node == nullptr) return kTfLiteError;
    const int* outputs;
    int num_outputs;
    TfLiteStatus tf_status = TfLiteOpaqueNodeOutputs(node, &outputs, &num_outputs);
    node_manager_->setOutputAtOperandIndex(outputs[0], result_node);
    return kTfLiteOk;
}

std::shared_ptr<OperationsBase> OpenVINOGraphBuilder::CreateOpClass(
    int operationIndex, TfLiteRegistrationExternal* registration) {
    switch (TfLiteRegistrationExternalGetBuiltInCode(registration)) {
        case kTfLiteBuiltinAdd: {
            return std::make_shared<Add>(operationIndex);
        }
        case kTfLiteBuiltinAveragePool2d: {
            return std::make_shared<AveragePool2D>(operationIndex);
        }
        case kTfLiteBuiltinConv2d: {
            return std::make_shared<Conv2D>(operationIndex);
        }
        case kTfLiteBuiltinConcatenation: {
            return std::make_shared<Concat>(operationIndex);
        }
        case kTfLiteBuiltinDepthwiseConv2d: {
            return std::make_shared<DepthwiseConv2D>(operationIndex);
            ;
        }
        case kTfLiteBuiltinDequantize: {
            return std::make_shared<Dequantize>(operationIndex);
            ;
	}
        case kTfLiteBuiltinMul: {
            return std::make_shared<Mul>(operationIndex);
        }
        case kTfLiteBuiltinResizeBilinear: {
            return std::make_shared<ResizeBilinear>(operationIndex);
            ;
        }
        case kTfLiteBuiltinRelu: {
            return std::make_shared<Relu>(operationIndex);
        }
        case kTfLiteBuiltinRelu6: {
            return std::make_shared<Relu6>(operationIndex);
        }
        case kTfLiteBuiltinLogistic: {
            return std::make_shared<Logistic>(operationIndex);
        }
        case kTfLiteBuiltinHardSwish: {
            return std::make_shared<HardSwish>(operationIndex);
        }
        case kTfLiteBuiltinSoftmax: {
            return std::make_shared<Softmax>(operationIndex);
        }
        default:
            return nullptr;
    }
}

}  // namespace openvinodelegate
}  // namespace tflite
