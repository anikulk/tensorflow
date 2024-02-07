#include "tensorflow/lite/delegates/openvino/operations/include/softmax.h"

namespace tflite {
namespace openvinodelegate {

std::shared_ptr<ov::Node> Softmax::CreateNode() {
    TfLiteSoftmaxParams* softmax_params = (TfLiteSoftmaxParams*)GetBuiltinData();
    auto input_node_1 = getInputNode(tensor_indices_[TFLITE_INPUT_NODE_1]);
    if (input_node_1 == nullptr) {
        TFLITE_LOG(INFO) << "input node 1 is null\n";
        return nullptr;
    }

    if (softmax_params->beta != 1.0f) {
        TFLITE_LOG(INFO) << "Unsupported Softmax op, beta value is not 1.0 \n";
        return nullptr;
    }

    input_node_1 = convertNHWCtoNCHW(TFLITE_INPUT_NODE_1, input_node_1);

    //NOTE: assumption here is: Tensorflow always computes softmax along channel(last) dimesnsion.
    //After transpose, our channel shifts to dim 1, which is default axis attribute for Softmax.
    auto softmax_node =
        std::make_shared<ov::opset8::Softmax>(input_node_1);
    return softmax_node;
}

}  // namespace openvinodelegate
}  // namespace tflite
