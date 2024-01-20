#include "tensorflow/lite/delegates/openvino/operations/include/conv2d.h"

namespace tflite {
namespace openvinodelegate {

std::shared_ptr<ov::Node> Conv2D::CreateNode() {
  const TfLiteConvParams* conv2d_params = (TfLiteConvParams*)GetBuiltinData();
  std::vector<int> filter_dims = GetDims(tensor_indices_[TFLITE_FILTER_NODE]);
  std::vector<size_t> strides;
  std::vector<std::ptrdiff_t> padding_begin, padding_end;
  std::vector<size_t> dilations;
  ov::op::PadType auto_pad;
  int filter_size = 0;
  int padding_top, padding_bottom, padding_left, padding_right = 0;

  if (conv2d_params->padding == kTfLitePaddingUnknown) {
  } else if (conv2d_params->padding == kTfLitePaddingSame) {
    auto_pad = ov::op::PadType::SAME_UPPER;
  } else if (conv2d_params->padding == kTfLitePaddingValid) {
    auto_pad = ov::op::PadType::VALID;
  }

  strides = {(size_t)conv2d_params->stride_height,
             (size_t)conv2d_params->stride_width};
  padding_begin = {padding_top, padding_left};
  padding_end = {padding_bottom, padding_right};
  dilations = {(size_t)conv2d_params->dilation_height_factor,
               (size_t)conv2d_params->dilation_width_factor};
  auto input_node = getInputNode(tensor_indices_[TFLITE_INPUT_NODE_1]);
  auto filter_node = getInputNode(tensor_indices_[TFLITE_FILTER_NODE]);
  auto bias_node = getInputNode(tensor_indices_[TFLITE_BIAS_NODE]);

  auto conv_node = std::make_shared<ov::opset8::Convolution>(
      input_node, filter_node, ov::Strides(strides),
      ov::CoordinateDiff(padding_begin), ov::CoordinateDiff(padding_end),
      ov::Strides(dilations), auto_pad);
  auto bias_dims = GetDims(tensor_indices_[TFLITE_BIAS_NODE]);
  std::vector<uint32_t> shape(conv_node->get_shape().size(), 1);
  shape[1] = bias_dims[0];
  auto shape_node =
      CreateConstNode(ov::element::i32, ov::Shape{shape.size()}, shape);

  bias_node =
      std::make_shared<ov::opset3::Reshape>(bias_node, shape_node, true);

  std::shared_ptr<ov::Node> outputNode = std::make_shared<ov::opset3::Add>(
      conv_node, bias_node, ov::op::AutoBroadcastType::NUMPY);

  return ApplyActivation(outputNode, conv2d_params->activation);

}

}  // namespace openvinodelegate
}  // namespace tflite
