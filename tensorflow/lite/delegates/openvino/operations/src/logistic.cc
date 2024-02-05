#include "tensorflow/lite/delegates/openvino/operations/include/logistic.h"

namespace tflite {
namespace openvinodelegate {

std::shared_ptr<ov::Node> Logistic::CreateNode() {
    // Creating input nodes
    auto input_node = getInputNode(tensor_indices_[TFLITE_INPUT_NODE_1]);
    if (input_node == nullptr) {
        return nullptr;
    }
    auto output_node = ApplyActivation(input_node, kTfLiteActSigmoid);
    return output_node;
}

}  // namespace openvinodelegate
}  // namespace tflite
