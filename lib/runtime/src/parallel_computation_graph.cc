#include "parallel_computation_graph.h"

namespace FlexFlow {

Operator const &
    ParallelComputationGraph::at(operator_guid_t const &guid) const {
  return this->graph.at(guid.value());
}

Operator &ParallelComputationGraph::at(operator_guid_t const &guid) {
  return this->graph.at(guid.value());
}

Operator const &
    ParallelComputationGraph::operator[](operator_guid_t const &guid) const {
  return this->graph.at(guid.value());
}

Operator &ParallelComputationGraph::at(operator_guid_t const &guid) {
  return this->graph.at(guid.value());
}

ParallelTensor const &
    ParallelComputationGraph::at(parallel_tensor_guid_t const &guid) const {
  return this->graph.at(guid.value());
}

ParallelTensor &
    ParallelComputationGraph::at(parallel_tensor_guid_t const &guid) {
  return this->graph.at(guid.value());
}

ParallelTensor const &ParallelComputationGraph::operator[](
    parallel_tensor_guid_t const &guid) const {
  return this->graph.at(guid.value());
}

ParallelTensor &ParallelCompuationGraph::operator[](
    parallel_tensor_guid_t const &guid) const {
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

OpTaskInvocation forward(ParallelComputationGraph const &pcg,
                         operator_guid_t const &op_guid) {
  return forward(pcg.at(op_guid));
}

OpTaskInvocation backward(ParallelComputationGraph const &pcg,
                          operator_guid_t const &op_guid) {
  return backward(pcg.at(op_guid));
}

ArgSpec resolve(ParallelComputationGraph const &pcg,
                operator_guid_t const &op_guid,
                OpArgRefSpec const &ref_spec) {
  OpArgRefType ref_type = ref_spec.get_ref_type();
  switch (ref_type) {
    default:
      throw mk_runtime_error("Unknown OpArgRefType {}", ref_type);
  }
}

struct ResolveOpArgSpec {
  ResolveOpArgSpec() = delete;
  ResolveOpArgSpec(ParallelComputationGraph const &pcg,
                   operator_guid_t const &op_guid)
      : pcg(pcg), op_guid(op_guid) {}

  ParallelComputationGraph const &pcg;
  operator_guid_t const &op_guid;

  ArgSpec operator()(OpArgSpec const &s) const {
    return resolve(pcg, op_guid, s);
  }

  template <typename T>
  ArgSpec operator()(T const &t) const {
    return t;
  }
};

ArgSpec resolve(ParallelComputationGraph const &pcg,
                operator_guid_t op_guid,
                OpArgSpec const &op_arg_spec) {
  return visit(ResolveOpArgSpec{pcg, op_guid}, op_arg_spec);
}

ParallelTensorSpec resolve(ParallelComputationGraph const &pcg,
                           operator_guid_t const &op_guid,
                           OpTensorSpec const &op_tensor_spec,
                           IsGrad const &is_grad) {
  optional<parallel_tensor_guid_t> pt_guid;
  switch (op_tensor_spec.role) {
    case TensorRole::INPUT:
      pt_guid = get_input_ptensors_by_idx(pcg, op_guid).at(op_tensor_spec.idx);
      break;
    case TensorRole::WEIGHT:
      pt_guid = get_weight_ptensors_by_idx(pcg, op_guid).at(op_tensor_spec.idx);
      break;
    case TensorRole::OUTPUT:
      pt_guid = get_output_ptensors_by_idx(pcg, op_guid).at(op_tensor_spec.idx);
      break;
    default:
      throw mk_runtime_error("Unknown tensor role {}", op_tensor_spec.role);
  }

  return {pt_guid.value(), is_grad};
}

TaskBinding resolve(ParallelComputationGraph const &pcg,
                    operator_guid_t const &op_guid,
                    OpTaskBinding const &op_binding) {
  TaskBinding binding(InvocationType::INDEX);

  for (auto const &kv : op_binding.get_tensor_bindings()) {
    slot_id slot = kv.first.first;
    IsGrad is_grad = kv.first.second;
    OpTensorSpec op_tensor_spec = kv.second;

    ParallelTensorSpec tensor_spec =
        resolve(pcg, op_guid, op_tensor_spec, is_grad);

    binding.bind(slot, tensor_spec);
  }

  for (auto const &kv : op_binding.get_arg_bindings()) {
    slot_id slot = kv.first;
    OpArgSpec op_arg_spec = kv.second;
  }
}

TaskInvocation resolve(ParallelComputationGraph const &pcg,
                       operator_guid_t,
                       OpTaskInvocation const &op_invocation) {
  return {op_invocation.task_id, resolve(pcg, op_invocation.binding)};
}

} // namespace FlexFlow
