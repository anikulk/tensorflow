#include "tensorflow/lite/delegates/openvino/operations/include/hardswish.h"

namespace tflite {
namespace openvinodelegate {

std::shared_ptr<ov::Node> HardSwish::CreateNode() {
    auto input_node = getInputNode(tensor_indices_[TFLITE_INPUT_NODE_1]);
    if (input_node == nullptr) {
        return nullptr;
    }
    auto output_node = std::make_shared<ov::op::v4::HSwish>(input_node);
    return output_node;
}

}  // namespace openvinodelegate
}  // namespace tflite
