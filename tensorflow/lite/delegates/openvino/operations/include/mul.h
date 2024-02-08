#ifndef TENSORFLOW_LITE_DELEGATES_OPENVINO_MUL_H_
#define TENSORFLOW_LITE_DELEGATES_OPENVINO_MUL_H_

#include "tensorflow/lite/delegates/openvino/operations/operations_base.h"

namespace tflite {
namespace openvinodelegate {

class Mul : public OperationsBase {
public:
    Mul(int operationIndex) {}
    std::shared_ptr<ov::Node> CreateNode() override;
};

}  // namespace openvinodelegate
}  // namespace tflite
#endif  // TENSORFLOW_LITE_DELEGATES_OPENVINO_MUL_H_
