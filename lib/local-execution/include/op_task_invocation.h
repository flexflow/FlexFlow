#ifndef _FLEXFLOW_RUNTIME_OP_TASK_SPEC_H
#define _FLEXFLOW_RUNTIME_OP_TASK_SPEC_H

#include "kernels/accessor.h"
#include "concrete_arg.h"
#include "op_arg_ref.h"
#include "op_task_signature.h"
#include "profiling.h"
#include "runtime_arg_ref.h"
#include "serialization.h"
#include "tasks.h"
#include "utils/bidict.h"
#include "utils/optional.h"
#include "utils/stack_map.h"
#include <variant>
#include <typeindex>
#include <unordered_map>
#include <unordered_set>

namespace FlexFlow {

enum class IsTrainable { YES, NO };

struct OpTensorSpec {
  TensorRole role;
  OpSlotOptions slot_option;
  req<int> idx;
};
FF_VISITABLE_STRUCT(OpTensorSpec, role, slot_option, idx);

OpTensorSpec input_tensor(int);
OpTensorSpec output_tensor(int);
OpTensorSpec weight_tensor(int);

using OpArgSpec = variant<ConcreteArgSpec,
                          OpArgRefSpec,
                          RuntimeArgRefSpec>;

struct OpArgSpecTypeAccessor {
  std::type_index operator()(OpArgSpec &spec) {
    return std::visit(
        [](auto &&arg) -> std::type_index {
          return arg.get_type_index();
        },
        spec);
  }
};

struct OpTaskBinding {
  OpTaskBinding() = default;

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
FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(OpTaskBinding, arg_bindings, tensor_bindings);

struct OpTaskInvocation {
public:
  OpTaskInvocation() = delete;
  OpTaskInvocation(task_id_t const &task_id, OpTaskBinding const &binding)
      : task_id(task_id), binding(binding) {}

public:
  task_id_t task_id;
  OpTaskBinding binding;
};
FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(OpTaskInvocation,
                                             task_id,
                                             binding);

OpTaskSignature infer_bwd_signature(OpTaskSignature const &fwd);
OpTaskBinding infer_bwd_binding(OpTaskBinding const &fwd);

bool validate_invocation(OpTaskSignature sig, OpTaskInvocation inv) {
  // tensors
  auto tensor_bindings = inv.binding.get_tensor_bindings();
  for (OpTensorSlotSpec const & op_tensor_slot_spec: sig.get_tensor_slots()) {
    slot_id name = op_tensor_slot_spec.name;
    IsGrad is_grad = op_tensor_slot_spec.is_grad;
    OpTensorSpec const & op_tensor_spec = tensor_bindings[std::make_pair<name, is_grad>()];
    if (op_tensor_spec.role != op_tensor_slot_spec.tensor_role || op_tensor_spec.slot_option != op_tensor_slot_spec.slot_option) {
      return false;
    }
  }

  // args
  auto sig_arg_types = sig.get_arg_types();
  for (auto arg_binding: inv.binding.get_arg_bindings()) {
    slot_id name = arg_binding.first;
    OpArgSpec op_arg_spec = arg_binding.second;
    std::type_index arg_type = sig_arg_types[name];
    if (OpArgSpecTypeAccessor(op_arg_spec) != arg_type) {
      return false;
    }
  }

  return true;
}

} // namespace FlexFlow

#endif
