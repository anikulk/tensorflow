#include "tensorflow/lite/delegates/openvino/operations/include/resize_bilinear.h"

namespace tflite {
namespace openvinodelegate {

std::shared_ptr<ov::Node> ResizeBilinear::createNode() {
    const TfLiteResizeBilinearParams* resize_bilinearParams =
        (TfLiteResizeBilinearParams*)GetBuiltinData();
    auto inputNode = getInputNode(tensor_indices[TFLITE_INPUT_NODE_1]);
    auto shapeNode = getInputNode(tensor_indices[TFLITE_INPUT_NODE_2]);
    struct ov::op::v11::Interpolate::InterpolateAttrs attrs;
    auto input_dims = GetDims(tensor_indices[TFLITE_INPUT_NODE_1]);
    auto shape_dims = GetDims(tensor_indices[TFLITE_INPUT_NODE_2]);

    attrs.mode = ov::op::v11::Interpolate::InterpolateMode::LINEAR_ONNX;
    attrs.shape_calculation_mode = ov::op::v11::Interpolate::ShapeCalcMode::SIZES;

    if (resize_bilinearParams->align_corners == true) {
        attrs.coordinate_transformation_mode =
            ov::op::v11::Interpolate::CoordinateTransformMode::ALIGN_CORNERS;
    } else if (resize_bilinearParams->half_pixel_centers == true) {
        attrs.coordinate_transformation_mode =
            ov::op::v11::Interpolate::CoordinateTransformMode::HALF_PIXEL;
    } else {
        attrs.coordinate_transformation_mode =
            ov::op::v11::Interpolate::CoordinateTransformMode::ASYMMETRIC;
    }

    std::vector<int32_t> axes_vec = {2, 3};
    auto axesNode = createConstNode(ov::element::i32, {2}, axes_vec);
    if (axesNode == nullptr) TFLITE_LOG(INFO) << "axes node is null \n";

    int32_t* size_data = new int32_t(2);
    GetTensorData(TFLITE_INPUT_NODE_2, (void*)size_data);
    std::vector<int32_t> size_vec = {size_data[0], size_data[1]};
    auto sizeNode = createConstNode(ov::element::i32, {2}, size_vec);
    if (sizeNode == nullptr) TFLITE_LOG(INFO) << "size node is null \n";

    auto outputNode =
        std::make_shared<ov::op::v11::Interpolate>(inputNode, sizeNode, axesNode, attrs);
    return outputNode;
}
}  // namespace openvinodelegate
}  // namespace tflite
