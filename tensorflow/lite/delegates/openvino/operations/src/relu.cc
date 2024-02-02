#include "tensorflow/lite/delegates/openvino/operations/include/relu.h"

namespace tflite {
namespace openvinodelegate {

std::shared_ptr<ov::Node> Relu::CreateNode() {
    auto input_node = getInputNode(tensor_indices_[TFLITE_INPUT_NODE_1]);
    if (input_node == nullptr) {
        return nullptr;
    }
    auto output_node = ApplyActivation(input_node, kTfLiteActRelu);
    return output_node;
}

}  // namespace openvinodelegate
}  // namespace tflite
