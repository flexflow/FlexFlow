#include "parallel_computation_graph.h"

namespace FlexFlow {

Operator const &ParallelComputationGraph::at(operator_guid_t guid) const {
  return this->graph.at(guid.value());
}

Operator &ParallelComputationGraph::at(operator_guid_t guid) {
  return this->graph.at(guid.value());
}

Operator const &ParallelComputationGraph::operator[](operator_guid_t guid) const {
  return this->graph.at(guid.value());
}

Operator &ParallelComputationGraph::at(operator_guid_t guid) {
  return this->graph.at(guid.value());
}

ParallelTensor const &ParallelComputationGraph::at(parallel_tensor_guid_t guid) const {
  return this->graph.at(guid.value());
}

ParallelTensor &ParallelComputationGraph::at(parallel_tensor_guid_t guid) {
  return this->graph.at(guid.value());
}

ParallelTensor const &ParallelComputationGraph::operator[](parallel_tensor_guid_t guid) const {
  return this->graph.at(guid.value());
}

ParallelTensor &ParallelCompuationGraph::operator[](parallel_tensor_guid_t guid) const {
  return this->graph.at(guid.value());
}

void swap(ParallelComputationGraph &lhs, ParallelComputationGraph &rhs) {
  swap(lhs.graph, rhs.graph);
}

struct Forward {
  template <typename T>
  OpTaskInvocation operator()(T const &t) {
    return forward(t);
  }
};

struct Backward {
  template <typename T>
  OpTaskInvocation operator()(T const &t) {
    return backward(t);
  }
};

OpTaskInvocation forward(PCGOperatorAttrs const &attrs) {
  return visit(Forward{}, attrs);
}

OpTaskInvocation backward(PCGOperatorAttrs const &attrs) {
  return visit(Backward{}, attrs);
}

OpTaskInvocation forward(ParallelComputationGraph const &pcg, operator_guid_t op_guid) {
  return forward(pcg.at(op_guid));
}

OpTaskInvocation backward(ParallelComputationGraph const &pcg, operator_guid_t op_guid) {
  return backward(pcg.at(op_guid));
}

TaskInvocation resolve_op_task_spec(ParallelComputationGraph const &pcg, OpTaskInvocation const &invocation) {

}

}
