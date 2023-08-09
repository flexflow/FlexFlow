#ifndef _FLEXFLOW_RUNTIME_OP_TASK_SPEC_H
#define _FLEXFLOW_RUNTIME_OP_TASK_SPEC_H

#include "accessor.h"
#include "legion.h"
#include "op_task_signature.h"
#include "profiling.h"
#include "runtime/config.h"
#include "serialization.h"
#include "task_invocation.h"
#include "tasks.h"
#include "utils/bidict.h"
#include "utils/optional.h"
#include "utils/stack_map.h"
#include <typeindex>
#include <unordered_map>
#include <unordered_set>

namespace FlexFlow {

enum class IsTrainable { YES, NO };

struct OpTensorSpec : public use_visitable_cmp<OpTensorSpec> {
public:
  OpTensorSpec() = delete;
  OpTensorSpec(TensorRole, int);

public:
  TensorRole role;
  int idx;
};

OpTensorSpec input_tensor(int);
OpTensorSpec output_tensor(int);
OpTensorSpec weight_tensor(int);

enum class OpArgRefType { PER_DEVICE_OP_STATE };

template <typename T>
struct OpArgRef : public use_visitable_cmp<OpArgRef<T>> {
public:
  OpArgRef() = delete;
  OpArgRef(OpArgRefType ref_type) : ref_type(ref_type) {}

public:
  OpArgRefType ref_type;
};

template <typename T>
OpArgRef<T> per_device_op_state() {
  return OpArgRef<T>(OpArgRefType::PER_DEVICE_OP_STATE);
}

struct OpArgRefSpec {
public:
  OpArgRefSpec() = delete;

  template <typename T>
  bool holds() const {
    return std::type_index(typeid(T)) == this->type;
  }

  OpArgRefType const &get_ref_type() const {
    return this->ref_type;
  }

  template <typename T>
  static OpArgRefSpec create(OpArgRef<T> const &r) {
    static_assert(is_serializable<T>, "Type must be serializable");

    return OpArgRefSpec(std::type_index(typeid(T)), r.ref_type);
  }

private:
  OpArgRefSpec(std::type_index, OpArgRefType);

  std::type_index type;
  OpArgRefType ref_type;
};

using OpArgSpec = variant<ConcreteArgSpec,
                          IndexArgSpec,
                          OpArgRefSpec,
                          CheckedTypedFuture,
                          CheckedTypedFutureMap,
                          ArgRefSpec,
                          TaskInvocationSpec>;

struct OpTaskBinding {
  OpTaskBinding() = default;

  static_assert(is_subeq_variant<ArgSpec, OpArgSpec>::value, "");

  void bind(slot_id, OpTensorSpec const &);
  void bind_grad(slot_id, OpTensorSpec const &);

  template <typename T>
  void bind_arg(slot_id name, T const &t) {
    this->insert_arg_spec(name, ConcreteArgSpec::create(t));
  }

  template <typename T>
  void bind_arg(slot_id name, OpArgRef<T> const &ref) {
    this->insert_arg_spec(name, OpArgRefSpec::create(ref));
  }

  template <typename T>
  void bind_arg(slot_id name, TypedFuture<T> const &f) {
    this->insert_arg_spec(name, CheckedTypedFuture::create(f));
  }

  template <typename T>
  void bind_arg(slot_id name, TypedFutureMap<T> const &fm) {
    this->insert_arg_spec(name, CheckedTypedFutureMap::create(fm));
  }

  std::unordered_map<std::pair<slot_id, IsGrad>, OpTensorSpec> const &
      get_tensor_bindings() const;
  std::unordered_map<slot_id, ArgSpec> const &get_arg_bindings() const;

private:
  void insert_arg_spec(slot_id name, ArgSpec const &arg_spec) {
    assert(!contains_key(this->arg_bindings, name));
    arg_bindings.insert({name, arg_spec});
  }

  // template <typename T>
  // ArgSpec generate_arg_spec(T const &t) {
  //   static_assert(is_serializable<T>, "Type must be serializable");

  //   size_t pre_size = serializer.get_used_bytes();
  //   ff_task_serialize(serializer, t);
  //   size_t post_size = serializer.get_used_bytes();
  //   return {
  //     typeid(T),
  //     pre_size,
  //     post_size - pre_size
  //   };
  // }

  /* Legion::Serializer serializer; */
  std::unordered_map<slot_id, ArgSpec> arg_bindings;
  std::unordered_map<std::pair<slot_id, IsGrad>, OpTensorSpec> tensor_bindings;
};

struct OpTaskInvocation : public use_visitable_cmp<OpTaskInvocation> {
public:
  OpTaskInvocation() = delete;
  OpTaskInvocation(task_id_t const &task_id, OpTaskBinding const &binding)
      : task_id(task_id), binding(binding) {}

public:
  task_id_t task_id;
  OpTaskBinding binding;
};

OpTaskSignature infer_bwd_signature(OpTaskSignature const &fwd);
OpTaskBinding infer_bwd_binding(OpTaskBinding const &fwd);

/* std::unordered_map<int, OpTensorSpec> get_regions_idxs(TaskArgumentFormat
 * const &); */

/* TaskArgumentFormat compile_task_invocation(OpTaskSignature const &,
 * OpTaskBinding const &); */

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::OpTensorSpec, role, idx);
VISITABLE_STRUCT(::FlexFlow::OpTensorSlotSpec, name, slot_type, tensor_role);

#endif
