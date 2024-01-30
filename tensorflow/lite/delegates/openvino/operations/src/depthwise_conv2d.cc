#include "tensorflow/lite/delegates/openvino/operations/include/depthwise_conv2d.h"

namespace tflite {
namespace openvinodelegate {

std::shared_ptr<ov::Node> DepthwiseConv2D::CreateNode() {
    const TfLiteDepthwiseConvParams* depth_conv2dParams =
        (TfLiteDepthwiseConvParams*)GetBuiltinData();
    // TODO: check for datatypes, tensor shapes, and non dynamic allocation
    auto input_node = getInputNode(tensor_indices_[TFLITE_INPUT_NODE_1]);
    auto filter_node = getInputNode(tensor_indices_[TFLITE_FILTER_NODE]);
    bool has_bias = false;
    ov::Output<ov::Node> bias_node;
    std::shared_ptr<ov::Node> output_node;
    std::vector<size_t> strides = {(size_t)depth_conv2dParams->stride_height,
                                   (size_t)depth_conv2dParams->stride_width};
    std::vector<size_t> dilations = {(size_t)depth_conv2dParams->dilation_height_factor,
                                     (size_t)depth_conv2dParams->dilation_width_factor};
    if (tensor_indices_[TFLITE_BIAS_NODE] < 0) {
        has_bias = false;
    } else {
        bias_node = getInputNode(tensor_indices_[TFLITE_BIAS_NODE]);
        has_bias = true;
    }

    std::string auto_pad;
    ov::op::PadType pad_type;
    auto input_dims = GetDims(tensor_indices_[TFLITE_INPUT_NODE_1]);

    TfLiteStatus status = CalculatePadding(depth_conv2dParams->padding, auto_pad);
    if (status != kTfLiteOk) {
        TFLITE_LOG(INFO) << "Invalid padding type in depthwise conv2d\n";
        return nullptr;
    }

    if (auto_pad == "same-upper") {
        pad_type = ov::op::PadType::SAME_UPPER;
    } else if (auto_pad == "valid") {
        pad_type = ov::op::PadType::VALID;
    }

    //TODO: lookout for order while running with an actual graph
    ov::AxisVector order = {3,0,1,2};
    const auto order_node = std::make_shared<ov::opset8::Constant>(
        ov::element::i64, ov::Shape{order.size()}, order);
    filter_node = std::make_shared<ov::opset3::Transpose>(filter_node, order_node);

    std::vector<size_t> shape(&filter_node->get_shape()[0], &filter_node->get_shape()[0] + 4);
    auto num_groups = input_dims[3] / filter_node->get_shape()[1];
    shape.insert(shape.begin(), num_groups);
    shape[1]  = filter_node->get_shape()[0] / num_groups;
    auto shape_node = CreateConstNode(ov::element::i32, ov::Shape{shape.size()}, shape);

    filter_node = std::make_shared<ov::opset3::Reshape>(filter_node, shape_node, true);

    auto depthwise_conv_node = std::make_shared<ov::opset3::GroupConvolution>(
        input_node, filter_node, ov::Strides(strides), ov::CoordinateDiff(0, 0),
        ov::CoordinateDiff(0, 0), ov::Strides(dilations), pad_type);

    if (has_bias) {
        auto bias_dimensions = GetDims(tensor_indices_[TFLITE_BIAS_NODE]);
        std::vector<uint32_t> shape(depthwise_conv_node->get_shape().size(), 1);
        shape[1] = bias_dimensions[0];
        auto shape_node = CreateConstNode(ov::element::i32, ov::Shape{shape.size()}, shape);
        bias_node = std::make_shared<ov::opset3::Reshape>(bias_node, shape_node, true);
        output_node = std::make_shared<ov::opset3::Add>(depthwise_conv_node, bias_node,
                                                       ov::op::AutoBroadcastType::NUMPY);
    } else {
        output_node = depthwise_conv_node;
    }

    output_node = ApplyActivation(output_node, depth_conv2dParams->activation);
    return output_node;
}

}  // namespace openvinodelegate
}  // namespace tflite
