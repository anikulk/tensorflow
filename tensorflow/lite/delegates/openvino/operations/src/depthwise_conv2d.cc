#include "tensorflow/lite/delegates/openvino/operations/include/depthwise_conv2d.h"

namespace tflite {
namespace openvinodelegate {

std::shared_ptr<ov::Node> DepthwiseConv2D::createNode() {
    const TfLiteDepthwiseConvParams* depth_conv2dParams = (TfLiteDepthwiseConvParams*)GetBuiltinData();
    //TODO: check for datatypes, tensor shapes, and non dynamic allocation
    auto inputNode = getInputNode(tensor_indices[TFLITE_INPUT_NODE_1]);
    auto filterNode = getInputNode(tensor_indices[TFLITE_FILTER_NODE]);
    bool has_bias = false;
    ov::Output<ov::Node> biasNode;
    std::vector<size_t> strides = {(size_t)depth_conv2dParams->stride_height, (size_t)depth_conv2dParams->stride_width};
    std::vector<size_t> dilations = {(size_t)depth_conv2dParams->dilation_height_factor, (size_t)depth_conv2dParams->dilation_width_factor};
    if (tensor_indices[TFLITE_BIAS_NODE] < 0) {
	has_bias = false;
    } else {
        biasNode = getInputNode(tensor_indices[TFLITE_BIAS_NODE]);
	has_bias = true;
    }

    int expected_height = 0;
    int expected_weight = 0;
    std::string auto_pad;
    auto pad_type = ov::op::PadType::EXPLICIT;
    std::vector<int32_t> padding(4, 0);
    auto filter_dims = GetDims(tensor_indices[TFLITE_FILTER_NODE]);
    auto input_dims = GetDims(tensor_indices[TFLITE_INPUT_NODE_1]);

    TfLiteStatus status = CalculatePadding(depth_conv2dParams->padding, auto_pad);
    if (status != kTfLiteOk) {
        TFLITE_LOG(INFO) << "Invalid padding type in depthwise conv2d\n";
	return nullptr;
    }

    if (auto_pad == "same-upper") {
        TfLitePaddingValues paddings = ComputePaddingHeightWidth(strides[0], strides[1], dilations[0], dilations[1], input_dims[1], input_dims[2], filter_dims[1], filter_dims[2], kTfLitePaddingSame, &expected_height, &expected_weight);
        padding[0] = paddings.height;// top
        padding[1] = paddings.height + paddings.height_offset; //bottom
        padding[2] = paddings.width; //left
        padding[3] = paddings.width + paddings.width_offset; //right
	pad_type = ov::op::PadType::SAME_UPPER;
    }

    //TODO: define strides, dilations, padding, auto_pad
    auto depthwise_convNode = std::make_shared<ov::opset3::GroupConvolution>(inputNode, filterNode, ov::Strides(strides), ov::CoordinateDiff(padding[0], padding[2]), ov::CoordinateDiff(padding[1], padding[3]), ov::Strides(dilations), pad_type);

    //TODO: update Bias if needed
    if(has_bias) {
        auto outputNode = std::make_shared<ov::opset8::Add>(depthwise_convNode, biasNode);
	return outputNode;
    } else
        return depthwise_convNode;
}

}  // namespace openvinodelegate
}  // namespace tflite
