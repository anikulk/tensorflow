#include "tensorflow/lite/delegates/openvino/operations/include/dequantize.h"

namespace tflite {
namespace openvinodelegate {

std::shared_ptr<ov::Node> Dequantize::CreateNode() {
  auto inputNode =
      getInputNode(tensor_indices_[0]);
  if (inputNode == nullptr)
      TFLITE_LOG(INFO) << "input node  is null\n";

  auto _shape = inputNode->get_shape();
            std::cout << "-+++++++++++++++++**********--dequantizeNode inputNode node shape--- " << _shape[0] << " "<<  _shape[1] << " "<< _shape[2] << " "<< _shape[3] << std::endl;

  auto dequantizeNode = std::make_shared<ov::opset8::Convert>(
        inputNode, ov::element::f32);

  _shape = dequantizeNode->get_shape();
            std::cout << "--+++++++++********************-dequantizeNode out node shape--------" << _shape[0] << " "<<  _shape[1] << " "<< _shape[2] << " "<< _shape[3] << std::endl;
  return dequantizeNode;
}

}  // namespace openvinodelegate
}  // namespace tflite