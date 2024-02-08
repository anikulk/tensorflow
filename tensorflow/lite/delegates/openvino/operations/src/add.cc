#include "tensorflow/lite/delegates/openvino/operations/include/add.h"

namespace tflite {
namespace openvinodelegate {

std::shared_ptr<ov::Node> Add::CreateNode() {
    TfLiteAddParams* add_params = (TfLiteAddParams*)GetBuiltinData();
    auto input_node_1 = getInputNode(tensor_indices_[TFLITE_INPUT_NODE_1]);
    if (input_node_1 == nullptr) {
        TFLITE_LOG(INFO) << "input node 1 is null\n";
        return nullptr;
    }
    auto input_node_2 = getInputNode(tensor_indices_[TFLITE_INPUT_NODE_2]);
    if (input_node_2 == nullptr) {
        TFLITE_LOG(INFO) << "input Node 2 is null\n";
        return nullptr;
    }

    input_node_1 = convertNHWCtoNCHW(TFLITE_INPUT_NODE_1, input_node_1);
    input_node_2 = convertNHWCtoNCHW(TFLITE_INPUT_NODE_2, input_node_2);

    auto add_node = std::make_shared<ov::opset8::Add>(input_node_1, input_node_2,
                                                      ov::op::AutoBroadcastType::NUMPY);
    auto output_node = ApplyActivation(add_node, add_params->activation);
    return output_node;
}

}  // namespace openvinodelegate
}  // namespace tflite
