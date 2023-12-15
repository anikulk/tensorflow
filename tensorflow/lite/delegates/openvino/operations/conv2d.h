#ifndef TENSORFLOW_LITE_DELEGATES_OPENVINO_CONV2D_H_
#define TENSORFLOW_LITE_DELEGATES_OPENVINO_CONV2D_H_

#include "operations_base.h"

namespace tflite {
namespace openvinodelegate {

class Conv2D : public OperationBuilder {
public:
    Conv2D(int operationIndex) { }
    std::shared_ptr<ov::Node> createNode() override;
};

}  // namespace openvinodelegate
}  // namespace tflite
#endif //TENSORFLOW_LITE_DELEGATES_OPENVINO_CONV2D_H_
