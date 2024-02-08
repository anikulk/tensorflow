#include "tensorflow/lite/delegates/openvino/operations/include/reshape.h"

namespace tflite {
namespace openvinodelegate {

std::shared_ptr<ov::Node> Reshape::CreateNode() {
    // arg - input node
    auto input_node = getInputNode(tensor_indices_[TFLITE_INPUT_NODE_1]);
    if (input_node == nullptr) {
        TFLITE_LOG(ERROR) << "input node is null\n";
        return nullptr;
    }

    // Validation check for input node dimensions
    auto input_node_dims = GetDims(tensor_indices_[TFLITE_INPUT_NODE_1]);
    for (int n : input_node_dims) {
        if (n == 0) {
            TFLITE_LOG(ERROR) << "zero input node dimension not allowed\n";
            // ToDo: returning nullptr causes seg fault. Handle ellegantly.
            // return nullptr;
        }
    }

    // shape_pattern - shape node
    ov::Output<ov::Node> shape_node = getInputNode(tensor_indices_[TFLITE_SHAPE_NODE]);

    // Validation check for shape rank
    auto shape_rank = shape_node.get_partial_shape().rank();
    auto shape_rank_length = shape_node.get_partial_shape().rank().get_length();
    if (shape_node.get_partial_shape().rank().get_length() > 1)
    {
        TFLITE_LOG(ERROR) << "shape must have rank 1 or be empty\n";
        // ToDo: returning nullptr causes seg fault. Handle ellegantly
        // return nullptr;
    }

    // special_zero
    // Set false since Keras doesn't have special_zero argument

    auto output_node = std::make_shared<ov::opset3::Reshape>(input_node, shape_node, false);
    if (output_node == nullptr) {
        TFLITE_LOG(ERROR) << "output node is null\n";
    }

    return output_node;
}

}  // namespace openvinodelegate
}  // namespace tflite