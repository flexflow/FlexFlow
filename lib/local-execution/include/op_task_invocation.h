#ifndef _FLEXFLOW_LOCAL_EXECUTION_OP_TASK_INVOCATION_H
#define _FLEXFLOW_LOCAL_EXECUTION_OP_TASK_INVOCATION_H

#include "concrete_arg.h"
#include "kernels/accessor.h"
#include "op_arg_ref.h"
#include "op_task_signature.h"
#include "op_tensor_spec.h"
#include "profiling.h"
#include "runtime_arg_ref.h"
#include "serialization.h"
#include "tasks.h"
#include "utils/bidict.h"
#include "utils/stack_map.h"
#include "variadic_tensor_ref.h"
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

  void bind(slot_id, VariadicTensorRef<OpTensorSpec> const &) {
    NOT_IMPLEMENTED();
  }
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

  void bind_args_from_fwd(OpTaskBinding const &fwd) {
    this->arg_bindings = fwd.get_arg_bindings();
  }

  void bind_tensors_from_fwd(OpTaskBinding const &fwd) {
    this->tensor_bindings = fwd.get_tensor_bindings();
  }

  std::unordered_map<std::pair<slot_id, IsGrad>, OpTensorSpec> const &
      get_tensor_bindings() const;
  std::unordered_map<slot_id, OpArgSpec> const &get_arg_bindings() const;

  void insert_arg_spec(slot_id name, OpArgSpec const &arg_spec) {
    assert(!contains_key(this->arg_bindings, name));
    this->arg_bindings.insert({name, arg_spec});
  }

  std::unordered_map<slot_id, OpArgSpec> arg_bindings;
  std::unordered_map<std::pair<slot_id, IsGrad>, OpTensorSpec> tensor_bindings;
};
FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(OpTaskBinding,
                                             arg_bindings,
                                             tensor_bindings);

struct OpTaskInvocation {
public:
  OpTaskInvocation() = delete;
  OpTaskInvocation(task_id_t const &task_id, OpTaskBinding const &binding)
      : task_id(task_id), binding(binding) {}

public:
  task_id_t task_id;
  OpTaskBinding binding;
};
FF_VISITABLE_STRUCT(OpTaskInvocation, task_id, binding);

OpTaskSignature infer_bwd_signature(OpTaskSignature const &fwd);
OpTaskBinding infer_bwd_binding(OpTaskBinding const &fwd);

bool is_invocation_valid(OpTaskSignature sig, OpTaskInvocation inv);

} // namespace FlexFlow

#endif
