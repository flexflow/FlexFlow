#ifndef _FLEXFLOW_RUNTIME_OP_TASK_SPEC_H
#define _FLEXFLOW_RUNTIME_OP_TASK_SPEC_H

#include "legion.h"
#include "tasks.h"
#include "utils/optional.h"
#include "runtime/config.h"
#include <unordered_set>
#include <unordered_map>
#include "utils/bidict.h"
#include "accessor.h"
#include "serialization.h"
#include <typeindex>
#include "utils/stack_map.h"
#include "accessor.h"
#include "task_spec.h"

namespace FlexFlow {

struct Op;

enum class TensorRole {
  INPUT,
  PARAM,
  OUTPUT,
};

enum class IsTrainable {
  YES,
  NO
};

enum class OpTaskType {
  INIT,
  FWD,
  BWD
};

struct OpTensorSpec : public use_visitable_cmp<OpTensorSpec> {
public:
  OpTensorSpec() = delete;
  OpTensorSpec(TensorRole, int, bool is_trainable, DataType, IsGrad is_grad = IsGrad::NO, optional<Legion::PrivilegeMode> mode = nullopt);
  OpTensorSpec(TensorRole, int, IsTrainable, DataType, IsGrad is_grad = IsGrad::NO, optional<Legion::PrivilegeMode> mode = nullopt);

  OpTensorSpec grad() const;

  Legion::PrivilegeMode get_privileges() const;
public:
  TensorRole role;
  int idx;
  IsGrad is_grad;
  IsTrainable is_trainable;
  DataType datatype;
  optional<Legion::PrivilegeMode> mode;
};

OpTensorSpec input_tensor(int, IsTrainable);
OpTensorSpec input_tensor(int, bool);

OpTensorSpec output_tensor(int);
OpTensorSpec param_tensor(int);

Legion::PrivilegeMode get_default_mode(OpTaskType, TensorRole, IsGrad);

struct OpTensorSlotSpec : public use_visitable_cmp<OpTensorSlotSpec> {
  OpTensorSlotSpec() = delete;
  OpTensorSlotSpec(slot_id, SlotType, TensorRole, IsGrad);

  slot_id name;
  SlotType slot_type;
  TensorRole tensor_role;
  IsGrad is_grad;

  Legion::PrivilegeMode get_privileges(OpTaskType) const;
};

OpTensorSlotSpec get_backward_slot(OpTensorSlotSpec const &forward_slot);
OpTensorSlotSpec get_backward_grad_slot(OpTensorSlotSpec const &forward_slot);

using ArgSlotSpec = std::type_index;

using OpSlotSpec = variant<OpTensorSlotSpec, ArgSlotSpec>;

bool is_tensor_slot(OpSlotSpec const &);
bool is_arg_slot(OpSlotSpec const &);
OpTensorSlotSpec get_tensor_slot(OpSlotSpec const &);
ArgSlotSpec get_arg_slot(OpSlotSpec const &);

struct OpTaskSignature {

};

struct OpTaskSignature {
  OpTaskSignature() = delete;
  OpTaskSignature(OpTaskType);

  OpTaskType get_task_type() const;

  void add_input_slot(slot_id);
  void add_input_slot(slot_id, SlotType);
  void add_input_slot(slot_id, Legion::PrivilegeMode);
  void add_input_slot(slot_id, SlotType, Legion::PrivilegeMode);

  void add_slot(OpSlotSpec const &);
  void add_slot(OpTensorSlotSpec const &);

  void add_slot(slot_id, SlotType, Legion::PrivilegeMode);
  void add_grad_slot(slot_id, SlotType, Legion::PrivilegeMode);

  void add_param_slot(slot_id, SlotType slot_type = SlotType::TENSOR);
  void add_output_slot(slot_id, SlotType slot_type = SlotType::TENSOR);

  void add_input_grad_slot(slot_id, SlotType slot_type = SlotType::TENSOR);
  void add_param_grad_slot(slot_id, SlotType slot_type = SlotType::TENSOR);
  void add_output_grad_slot(slot_id, SlotType slot_type = SlotType::TENSOR);

  bool operator==(OpTaskSignature const &) const;
  bool operator!=(OpTaskSignature const &) const;
  template <typename T>
  void add_arg_slot(slot_id name) {
    static_assert(is_serializable<T>, "Type must be serializable");

  std::unordered_set<OpSlotSpec> get_slots() const;

  OpSlotSpec get_slot(slot_id) const;
private:
  std::unordered_map<slot_id, std::type_index> task_arg_types;
  std::unordered_map<slot_id, TensorRole> slots;
};

using SlotKey = std::pair<slot_id, IsGrad>;

struct OpTaskBinding {
  OpTaskBinding() {
    serializer.reserve_bytes(sizeof(TaskArgumentFormat));
  }

  void bind(slot_id, OpTensorSpec const &);
  void bind_grad(slot_id, OpTensorSpec const &);

  template <typename T>
  void bind_arg(slot_id name, T const &t) {
    auto arg_spec = this->generate_arg_spec<T>(t);
    assert (!contains_key(this->arg_bindings, name));
    arg_bindings.insert({name, arg_spec});
  }

  void bind(std::vector<std::pair<slot_id, OpTensorSpec>> const &);

  std::unordered_map<slot_id, OpTensorSpec> const &get_tensor_bindings() const;
  std::unordered_map<slot_id, ArgSpec> const &get_arg_bindings() const;

  Legion::TaskArgument get_legion_task_arg() const;
private:
  template <typename T>
  ArgSpec generate_arg_spec(T const &t) {
    static_assert(is_serializable<T>, "Type must be serializable");

    size_t pre_size = serializer.get_used_bytes();
    ff_task_serialize(serializer, t);
    size_t post_size = serializer.get_used_bytes();
    return {
      typeid(T),
      pre_size,
      post_size - pre_size
    };
  }

  Legion::Serializer serializer;
  std::unordered_map<slot_id, ArgSpec> arg_bindings;
  std::unordered_map<slot_id, OpTensorSpec> bindings;

  friend TaskArgumentFormat compile_task_invocation(OpTaskSignature const &, OpTaskBinding &);
};

OpTaskSignature infer_bwd_signature(OpTaskSignature const &fwd);
OpTaskBinding infer_bwd_binding(OpTaskBinding const &fwd);

std::unordered_map<int, OpTensorSpec> get_regions_idxs(TaskArgumentFormat const &);

TaskArgumentFormat compile_task_invocation(OpTaskSignature const &, OpTaskBinding const &);

}

VISITABLE_STRUCT(::FlexFlow::OpTensorSpec, role, idx, is_grad, is_trainable, mode);
VISITABLE_STRUCT(::FlexFlow::OpTensorSlotSpec, name, slot_type, tensor_role, is_grad);


#endif
