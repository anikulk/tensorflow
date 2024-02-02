#ifndef TENSORFLOW_LITE_DELEGATES_OPENVINO_RELU6_H_
#define TENSORFLOW_LITE_DELEGATES_OPENVINO_RELU6_H_

#include "tensorflow/lite/delegates/openvino/operations/operations_base.h"

namespace tflite {
namespace openvinodelegate {

class Relu6 : public OperationsBase {
public:
    Relu6(int operationIndex) {}
    std::shared_ptr<ov::Node> CreateNode() override;
};

}  // namespace openvinodelegate
}  // namespace tflite
#endif  // TENSORFLOW_LITE_DELEGATES_OPENVINO_RELU6_H_
