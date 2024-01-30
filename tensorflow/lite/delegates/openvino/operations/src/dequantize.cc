#include "tensorflow/lite/delegates/openvino/operations/include/dequantize.h"

namespace tflite {
namespace openvinodelegate {

std::shared_ptr<ov::Node> Dequantize::CreateNode() {
  auto inputNode =
      getInputNode(tensor_indices_[0]);
  if (inputNode == nullptr)
      TFLITE_LOG(INFO) << "input node  is null\n";

  auto dequantizeNode = std::make_shared<ov::opset8::Convert>(
        inputNode, ov::element::f32);
        
  return dequantizeNode;
}

}  // namespace openvinodelegate
}  // namespace tflite