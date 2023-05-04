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
#include "profiling.h"

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
  OpTensorSpec(TensorRole, int);

  OpTensorSpec grad() const;
public:
  TensorRole role;
  int idx;
};

OpTensorSpec input_tensor(int);
OpTensorSpec output_tensor(int);
OpTensorSpec param_tensor(int);

template <typename T>
struct OpArgRef : public use_visitable_cmp<OpArgRef<T>> {
};

OpArgRef<EnableProfiling> enable_profiling();
OpArgRef<PerDeviceFFHandle> ff_handle();
OpArgRef<PerDeviceOpState *> per_device_op_state();

struct OpTensorSlotSpec : public use_visitable_cmp<OpTensorSlotSpec> {
public:
  OpTensorSlotSpec() = delete;
  OpTensorSlotSpec(slot_id, SlotType, TensorRole);

public:
  slot_id name;
  SlotType slot_type;
  TensorRole tensor_role;
};

struct OpTaskSignature {
  OpTaskSignature() = delete;
  OpTaskSignature(OpTaskType);

  OpTaskType get_task_type() const;

  void add_input_slot(slot_id, SlotType slot_type = SlotType::TENSOR);
  void add_output_slot(slot_id, SlotType slot_type = SlotType::TENSOR);
  void add_weight_slot(slot_id, SlotType slot_type = SlotType::TENSOR);

  /* void add_input_slot(slot_id, Legion::PrivilegeMode); */
  /* void add_input_slot(slot_id, SlotType, Legion::PrivilegeMode); */

  bool operator==(OpTaskSignature const &) const;
  bool operator!=(OpTaskSignature const &) const;

  template <typename T>
  void add_arg_slot(slot_id name) {
    static_assert(is_serializable<T>, "Type must be serializable");
  }

private:
  std::unordered_map<slot_id, std::type_index> task_arg_types;
  std::unordered_map<slot_id, TensorRole> slots;
};

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

  template <typename T> 
  void bind_arg(slot_id name, OpArgRef<T> const &ref);

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

struct OpTaskInvocation : public use_visitable_cmp<OpTaskInvocation> {
public:
  OpTaskInvocation() = delete;
  OpTaskInvocation(task_id_t const &task_id, OpTaskBinding const &binding)
    : task_id(task_id), binding(binding) { }

public:
  task_id_t task_id;
  OpTaskBinding binding;
};

OpTaskSignature infer_bwd_signature(OpTaskSignature const &fwd);
OpTaskBinding infer_bwd_binding(OpTaskBinding const &fwd);

std::unordered_map<int, OpTensorSpec> get_regions_idxs(TaskArgumentFormat const &);

TaskArgumentFormat compile_task_invocation(OpTaskSignature const &, OpTaskBinding const &);

template <task_id_t> OpTaskSignature get_signature();

template <typename F>
void register_task(task_id_t, std::string const &name, OpTaskSignature const &, F const &func);

template <typename F>
void register_task(task_id_t, std::string const &name, OpTaskSignature const &, F const &func, F const &cpu_func);

}

VISITABLE_STRUCT(::FlexFlow::OpTensorSpec, role, idx);
VISITABLE_STRUCT(::FlexFlow::OpTensorSlotSpec, name, slot_type, tensor_role);


#endif
