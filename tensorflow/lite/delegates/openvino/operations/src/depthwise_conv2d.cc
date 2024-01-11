#include "tensorflow/lite/delegates/openvino/operations/include/depthwise_conv2d.h"

namespace tflite {
namespace openvinodelegate {

std::shared_ptr<ov::Node> DepthwiseConv2D::CreateNode() {
    const TfLiteDepthwiseConvParams* depth_conv2dParams = (TfLiteDepthwiseConvParams*)GetBuiltinData();
    //TODO: check for datatypes, tensor shapes, and non dynamic allocation
    auto inputNode = getInputNode(tensor_indices_[TFLITE_INPUT_NODE_1]);
    auto filterNode = getInputNode(tensor_indices_[TFLITE_FILTER_NODE]);
    bool has_bias = false;
    ov::Output<ov::Node> biasNode;
    std::shared_ptr<ov::Node> outputNode;
    std::vector<size_t> strides = {(size_t)depth_conv2dParams->stride_height, (size_t)depth_conv2dParams->stride_width};
    std::vector<size_t> dilations = {(size_t)depth_conv2dParams->dilation_height_factor, (size_t)depth_conv2dParams->dilation_width_factor};
    if (tensor_indices_[TFLITE_BIAS_NODE] < 0) {
	has_bias = false;
    } else {
        biasNode = getInputNode(tensor_indices_[TFLITE_BIAS_NODE]);
	has_bias = true;
    }

    int expected_height = 0;
    int expected_weight = 0;
    std::string auto_pad;
    auto pad_type = ov::op::PadType::EXPLICIT;
    std::vector<int32_t> padding(4, 0);
    auto filter_dims = GetDims(tensor_indices_[TFLITE_FILTER_NODE]);
    auto input_dims = GetDims(tensor_indices_[TFLITE_INPUT_NODE_1]);

    TfLiteStatus status = CalculatePadding(depth_conv2dParams->padding, auto_pad);
    if (status != kTfLiteOk) {
        TFLITE_LOG(INFO) << "Invalid padding type in depthwise conv2d\n";
	return nullptr;
    }

    if (auto_pad == "same-upper") {
        padding[0] = 0;// top
        padding[1] = 0; //bottom
        padding[2] = 0; //left
        padding[3] = 0; //right
	pad_type = ov::op::PadType::SAME_UPPER;
    }
    //TODO: define paddings for explicit padding

    auto depthwise_convNode = std::make_shared<ov::opset3::GroupConvolution>(inputNode, filterNode, ov::Strides(strides), ov::CoordinateDiff(padding[0], padding[2]), ov::CoordinateDiff(padding[1], padding[3]), ov::Strides(dilations), pad_type);

    if(has_bias) {
        auto biasDimensions = GetDims(tensor_indices_[TFLITE_BIAS_NODE]);
        std::vector<uint32_t> shape(depthwise_convNode->get_shape().size(), 1);
        shape[1] = biasDimensions[0];
	auto shapeNode = CreateConstNode(ov::element::i32, ov::Shape{shape.size()}, shape);
	biasNode = std::make_shared<ov::opset3::Reshape>(biasNode, shapeNode, true);
	outputNode = std::make_shared<ov::opset3::Add>(
			depthwise_convNode, biasNode, ov::op::AutoBroadcastType::NUMPY);
    } else {
        outputNode = depthwise_convNode;
    }

    outputNode = ApplyActivation(outputNode, depth_conv2dParams->activation);
}

}  // namespace openvinodelegate
}  // namespace tflite
