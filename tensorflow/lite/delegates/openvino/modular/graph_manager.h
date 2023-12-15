#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace openvinodelegate {

class GraphExecutionManagerInterface {
  TfLiteStatus initializeGraph();
  TfLiteStatus generateGraph();
  void createInputParams();
private:
  GraphBuilder* graphBuilder;
  OperationBuilder* opBuilder;
  void instantitateOpClassFactory();
  std::vector<std::shared_ptr<OperationsBase>> opNodes;
  TfLiteContent* context;
  const TfLiteDelegateParams* params;
  std::vector<int> tensors_to_replace;
};

class OpenVINOGraphExecutionManager : public GraphExecutionManagerInterface {
    GraphBuilder* graphBuilder; // use smart pointer?
    OperationBuilder* opBuilder;
}

} //namespace openvinodelegate
} //namespace tflite
