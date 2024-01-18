#ifndef TENSORFLOW_LITE_DELEGATES_OPENVINO_NODE_MANAGER_H_
#define TENSORFLOW_LITE_DELEGATES_OPENVINO_NODE_MANAGER_H_

#include <openvino/openvino.hpp>

class NodeManager {
public:
    NodeManager() {}
    std::shared_ptr<ov::Node> getInterimNodeOutput(int index) {
        auto node = output_at_op_index_[index];
        return node.get_node_shared_ptr();
    }
    void setOutputAtOperandIndex(int index, ov::Output<ov::Node> output) {
        output_at_op_index_.insert(std::pair<int, ov::Output<ov::Node>>(index, output));
    }

    size_t getNodeCount() const { return output_at_op_index_.size(); }

private:
    std::map<int, ov::Output<ov::Node>> output_at_op_index_;
};
#endif  // TENSORFLOW_LITE_DELEGATES_OPENVINO_NODE_MANAGER_H_
