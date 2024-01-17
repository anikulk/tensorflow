#include "tensorflow/lite/delegates/openvino/operations/include/depthwise_conv2d.h"

namespace tflite {
namespace openvinodelegate {

std::shared_ptr<ov::Node> DepthwiseConv2D::createNode() {
    const TfLiteDepthwiseConvParams* depth_conv2dParams =
        (TfLiteDepthwiseConvParams*)GetBuiltinData();
    // TODO: check for datatypes, tensor shapes, and non dynamic allocation
    auto inputNode = getInputNode(tensor_indices[TFLITE_INPUT_NODE_1]);
    auto filterNode = getInputNode(tensor_indices[TFLITE_FILTER_NODE]);
    bool has_bias = false;
    ov::Output<ov::Node> biasNode;
    std::shared_ptr<ov::Node> outputNode;
    std::vector<size_t> strides = {(size_t)depth_conv2dParams->stride_height,
                                   (size_t)depth_conv2dParams->stride_width};
    std::vector<size_t> dilations = {(size_t)depth_conv2dParams->dilation_height_factor,
                                     (size_t)depth_conv2dParams->dilation_width_factor};
    if (tensor_indices[TFLITE_BIAS_NODE] < 0) {
        has_bias = false;
    } else {
        biasNode = getInputNode(tensor_indices[TFLITE_BIAS_NODE]);
        has_bias = true;
    }

    std::string auto_pad;
    ov::op::PadType pad_type;
    auto input_dims = GetDims(tensor_indices[TFLITE_INPUT_NODE_1]);

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
    ov::AxisVector order = {1,0,2,3};
    const auto order_node = std::make_shared<ov::opset8::Constant>(
        ov::element::i64, ov::Shape{order.size()}, order);
    filterNode = std::make_shared<ov::opset3::Transpose>(filterNode, order_node);

    std::vector<size_t> shape(&filterNode->get_shape()[0], &filterNode->get_shape()[0] + 4);
    auto num_groups = input_dims[3] / filterNode->get_shape()[1];
    shape.insert(shape.begin(), num_groups);
    shape[1]  = filterNode->get_shape()[0] / num_groups;
    auto shapeNode = createConstNode(ov::element::i32, ov::Shape{shape.size()}, shape);

    filterNode = std::make_shared<ov::opset3::Reshape>(filterNode, shapeNode, true);

    auto depthwise_convNode = std::make_shared<ov::opset3::GroupConvolution>(
        inputNode, filterNode, ov::Strides(strides), ov::CoordinateDiff(0, 0),
        ov::CoordinateDiff(0, 0), ov::Strides(dilations), pad_type);

    if (has_bias) {
        auto biasDimensions = GetDims(tensor_indices[TFLITE_BIAS_NODE]);
        std::vector<uint32_t> shape(depthwise_convNode->get_shape().size(), 1);
        shape[1] = biasDimensions[0];
        auto shapeNode = createConstNode(ov::element::i32, ov::Shape{shape.size()}, shape);
        biasNode = std::make_shared<ov::opset3::Reshape>(biasNode, shapeNode, true);
        outputNode = std::make_shared<ov::opset3::Add>(depthwise_convNode, biasNode,
                                                       ov::op::AutoBroadcastType::NUMPY);
    } else {
        outputNode = depthwise_convNode;
    }

    outputNode = ApplyActivation(outputNode, depth_conv2dParams->activation);
    return outputNode;
}

}  // namespace openvinodelegate
}  // namespace tflite
