#include "openvino_delegate_kernel.h"

namespace tflite {
namespace openvinodelegate {
class OpenVINODelegateKernel : public SimpleDelegateKernelInterface {
  TfLiteStatus Init(TfLiteContext* context,
                            const TfLiteDelegateParams* params) override {
        for (int i = 0; i < params->nodes_to_replace->size; i++) {
            const int node_id = params->nodes_to_replace->data[i];
            TfLiteNode* delegate_node;
            TfLiteRegistration* delegate_node_registration;
            GetNodeAndRegistration(context, node_id, &delegate_node, 
                                    &delegate_node_registration);
            std::vector<int> node_inputs;
            node_inputs.resize(delegate_node->inputs->size);
            for (int j = 0; j < delegate_node->size; j++) {
                node_inputs.pushback(delegate_node->data[j]);
            }
            input_index_map.insert(std::pair(i, node_inputs));
        }
  }

  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override {

  }

  TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override {

  }

  // Map with graph node index as key and vector of input indices at node in key. 
  std::map<int, std::vector<int>> input_index_map;

  // Map with graph node index as key and vector of output indices at node in key. 
  std::map<int, std::vector<int>> output_index_map;

};
}
}
