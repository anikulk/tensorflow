#include "tensorflow/lite/delegates/openvino/operations/include/resize_bilinear.h"

namespace tflite {
namespace openvinodelegate {

std::shared_ptr<ov::Node> ResizeBilinear::CreateNode() {
    std::cout << "---create node ResizeBilinear Enter-----" << std::endl;
    const TfLiteResizeBilinearParams* resize_bilinearParams =
        (TfLiteResizeBilinearParams*)GetBuiltinData();
    auto input_node = getInputNode(tensor_indices_[TFLITE_INPUT_NODE_1]);
    auto shape_node = getInputNode(tensor_indices_[TFLITE_INPUT_NODE_2]);
    struct ov::op::v11::Interpolate::InterpolateAttrs attrs;
    auto input_dims = GetDims(tensor_indices_[TFLITE_INPUT_NODE_1]);
    auto shape_dims = GetDims(tensor_indices_[TFLITE_INPUT_NODE_2]);

    auto shape = input_node->get_shape();
    std::cout << "-input shape----- " << shape[0] << " " << shape[1] << " " << shape[2] << " " << shape[3] << std::endl;

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
    auto axes_node = CreateConstNode(ov::element::i32, {2}, axes_vec);
    if (axes_node == nullptr) TFLITE_LOG(INFO) << "axes node is null \n";
    std::cout << "---create node ResizeBilinear Axis-----" << std::endl;

    int32_t* size_data = new int32_t(2);
    GetTensorData(tensor_indices_[TFLITE_INPUT_NODE_2], (void*)size_data);
    std::vector<int32_t> size_vec = {size_data[0], size_data[1]};
    std::cout << "-size data----- " << size_data[0] << " " << size_data[1] << std::endl;
    auto size_node = CreateConstNode(ov::element::i32, {2}, size_vec);
    if (size_node == nullptr) TFLITE_LOG(INFO) << "size node is null \n";
    std::cout << "---create node ResizeBilinear size-----" << std::endl;

    delete(size_data);
    auto output_node =
        std::make_shared<ov::op::v11::Interpolate>(input_node, size_node, axes_node, attrs);
    
    

    std::cout << "---create node ResizeBilinear Exit-----------" << std::endl;
    return output_node;
}
}  // namespace openvinodelegate
}  // namespace tflite
