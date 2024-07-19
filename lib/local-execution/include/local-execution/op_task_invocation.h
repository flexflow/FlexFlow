#ifndef _FLEXFLOW_LOCAL_EXECUTION_OP_TASK_INVOCATION_H
#define _FLEXFLOW_LOCAL_EXECUTION_OP_TASK_INVOCATION_H

#include "kernels/accessor.h"
#include "local-execution/concrete_arg.h"
#include "local-execution/op_arg_ref.h"
#include "local-execution/op_task_signature.h"
#include "local-execution/op_tensor_spec.h"
#include "local-execution/profiling.h"
#include "local-execution/runtime_arg_ref.h"
#include "local-execution/tasks.h"
#include "local-execution/variadic_tensor_ref.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "pcg/computation_graph.h"
#include "utils/bidict/bidict.h"
#include "utils/stack_map.h"
#include <typeindex>
#include <unordered_map>
#include <unordered_set>
#include <variant>

namespace FlexFlow {

enum class IsTrainable { YES, NO };

using OpArgSpec =
    std::variant<ConcreteArgSpec, OpArgRefSpec, RuntimeArgRefSpec>;

struct OpTaskBinding {
  OpTaskBinding() = default;

  void bind(slot_id, VariadicTensorRef<OpTensorSpec> const &);
  void bind(slot_id, OpTensorSpec const &);
  void bind_grad(slot_id, OpTensorSpec const &);

  template <typename T>
  void bind_device_specific_arg(slot_id name, T const &t) {
    NOT_IMPLEMENTED();
  }

  template <typename T>
  void bind_device_specific_arg(slot_id name, OpArgRef<T> const &t) {
    NOT_IMPLEMENTED();
  }

  template <typename T>
  void bind_arg(slot_id name, T const &t) {
    this->insert_arg_spec(name, ConcreteArgSpec::create(t));
  }

  template <typename T>
  void bind_arg(slot_id name, RuntimeArgRef<T> const &ref) {
    this->insert_arg_spec(name, RuntimeArgRefSpec::create(ref));
  }

  template <typename T>
  void bind_arg(slot_id name, OpArgRef<T> const &ref) {
    this->insert_arg_spec(name, OpArgRefSpec::create(ref));
  }

  std::unordered_map<std::pair<slot_id, IsGrad>, OpTensorSpec> const &
      get_tensor_bindings() const;
  std::unordered_map<slot_id, OpArgSpec> const &get_arg_bindings() const;

  void bind_from_forward(OpTaskBinding const &fwd);

private:
  void insert_arg_spec(slot_id name, OpArgSpec const &arg_spec);
  std::unordered_map<std::pair<slot_id, IsGrad>, OpTensorSpec> tensor_bindings;
  std::unordered_map<slot_id, OpArgSpec> arg_bindings;
};

struct OpTaskInvocation {
public:
  OpTaskInvocation() = delete;
  OpTaskInvocation(task_id_t const &task_id, OpTaskBinding const &binding)
      : task_id(task_id), binding(binding) {}

public:
  task_id_t task_id;
  OpTaskBinding binding;
};

OpTaskInvocation init(ComputationGraphOpAttrs const &);
OpTaskInvocation forward(ComputationGraphOpAttrs const &);
OpTaskInvocation backward(ComputationGraphOpAttrs const &);

OpTaskSignature infer_bwd_signature(OpTaskSignature const &fwd);
OpTaskBinding infer_bwd_binding(OpTaskBinding const &fwd);

bool is_invocation_valid(OpTaskSignature const &sig,
                         OpTaskInvocation const &inv);

} // namespace FlexFlow

#endif
