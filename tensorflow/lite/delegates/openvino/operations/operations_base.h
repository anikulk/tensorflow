#ifndef TENSORFLOW_LITE_DELEGATES_OPERATIONS_BASE_H_
#define TENSORFLOW_LITE_DELEGATES_OPERATIONS_BASE_H_

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset3.hpp>
#include <openvino/opsets/opset8.hpp>
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/openvino/operations/openvino_node_manager.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/tools/logging.h"

#define TFLITE_INPUT_NODE_1 0
#define TFLITE_INPUT_NODE_2 1
#define TFLITE_FILTER_NODE 1
#define TFLITE_BIAS_NODE 2

namespace tflite {
namespace openvinodelegate {

class OperationsBase {
public:
    void UpdateNodeInfo(void* data, int size, void* builtin_data) {
        tensor_indices_ = (int*)data;
        tensor_indices_size_ = size;
        SetBuiltinData(builtin_data);
    }
    void SetGraphData(const TfLiteOpaqueContext* context, NodeManager* node_manager) {
        context_ = context;
        node_manager_ = node_manager;
    }
    virtual std::shared_ptr<ov::Node> CreateNode() = 0;

protected:
    // tflite runtime related info to be added in Model BUilder
    int operation_index_;
    void* GetBuiltinData() { return builtin_data_; }
    void SetBuiltinData(void* builtin_data) { builtin_data_ = builtin_data; }
    std::shared_ptr<ov::Node> getInputNode(int index) {
        return node_manager_->getInterimNodeOutput(index);
    }

    template <typename T>
    std::shared_ptr<ov::Node> CreateConstNode(ov::element::Type elementType, ov::Shape shape,
                                              std::vector<T> data) {
        return std::make_shared<ov::opset8::Constant>(elementType, shape, data);
    }

    TfLiteStatus CalculatePadding(TfLitePadding padding, std::string& auto_pad) {
        switch (padding) {
            case kTfLitePaddingSame: {
                auto_pad = "same-upper";
                return kTfLiteOk;
            }
            case kTfLitePaddingValid: {
                auto_pad = "valid";
                return kTfLiteOk;
            }
            default:
                return kTfLiteError;
        }
    }

    std::shared_ptr<ov::Node> ApplyActivation(std::shared_ptr<ov::Node> input,
                                              TfLiteFusedActivation activation) {
        // TODO: change activation type from Tflite to OV runtime
        switch (activation) {
            case kTfLiteActNone:
                return input;
            case kTfLiteActRelu:
                return std::make_shared<ov::opset8::Relu>(input);
            case kTfLiteActReluN1To1:
                return std::make_shared<ov::opset8::Clamp>(input, -1, 1);
            case kTfLiteActRelu6:
                return std::make_shared<ov::opset8::Clamp>(input, 0, 6);
            case kTfLiteActTanh:
                return std::make_shared<ov::opset8::Tanh>(input);
            case kTfLiteActSignBit:
                return nullptr;
            case kTfLiteActSigmoid:
                return std::make_shared<ov::opset8::Sigmoid>(input);
            default:
                return nullptr;
        }
    }

    std::vector<int> GetDims(int index) {
        auto t = TfLiteOpaqueContextGetOpaqueTensor(context_, index);
        int32_t num_dims;
        num_dims = TfLiteOpaqueTensorNumDims(t);
        std::vector<int> dims(num_dims);
        for (int i = 0; i < num_dims; i++) {
            dims[i] = TfLiteOpaqueTensorDim(t, i);
        }
        return dims;
    }

    void GetTensorData(int index, void* data) {
        auto opaque_tensor = TfLiteOpaqueContextGetOpaqueTensor(context_, index);
        void* tensor_data = TfLiteOpaqueTensorData(opaque_tensor);
        auto size = TfLiteOpaqueTensorByteSize(opaque_tensor);
        std::memcpy(data, tensor_data, size);
    }

    std::shared_ptr<ov::Node> convertNHWCtoNCHW(int index, std::shared_ptr<ov::Node> input) {
        auto node_dims = GetDims(tensor_indices_[index]);
        ov::AxisVector order = {0, 3, 1, 2};
        const auto order_node = std::make_shared<ov::opset8::Constant>(
            ov::element::i64, ov::Shape{order.size()}, order);
        if (node_dims.size() < 4 && node_dims.size() > 0) {
            auto size = node_dims.size();
            for (int i = 0; i < 4 - size; i++) {
                node_dims.insert(node_dims.begin(), 1);
            }
            auto new_size = CreateConstNode(ov::element::i32, ov::Shape{4}, node_dims);
            input = std::make_shared<ov::opset8::Reshape>(input, new_size, false);
            input = std::make_shared<ov::opset3::Transpose>(input, order_node);
        }
        if (node_dims.size() == 5) {
            order = {0, 4, 1, 2, 3};
            const auto order_node = std::make_shared<ov::opset8::Constant>(
                ov::element::i64, ov::Shape{order.size()}, order);
            input = std::make_shared<ov::opset3::Transpose>(input, order_node);
        }
        return input;
    }

    int* tensor_indices_;
    int tensor_indices_size_;

private:
    void* builtin_data_ = nullptr;
    int op_type_ = 0;
    NodeManager* node_manager_;
    const TfLiteOpaqueContext* context_;
};

}  // namespace openvinodelegate
}  // namespace tflite
#endif  // TENSORFLOW_LITE_DELEGATES_OPERATIOSN_BASE_H_
