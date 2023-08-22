#ifndef _FLEXFLOW_RUNTIME_OP_TASK_SPEC_H
#define _FLEXFLOW_RUNTIME_OP_TASK_SPEC_H

#include "accessor.h"
#include "index_task_invocation.h"
#include "legion.h"
#include "op_arg_ref.h"
#include "op_task_signature.h"
#include "runtime/config.h"
#include "runtime/profiling.h"
#include "serialization.h"
#include "standard_task_invocation.h"
#include "tasks.h"
#include "utils/bidict.h"
#include "utils/optional.h"
#include "utils/stack_map.h"
#include <typeindex>
#include <unordered_map>
#include <unordered_set>

namespace FlexFlow {

enum class IsTrainable { YES, NO };

struct OpTensorSpec {
  TensorRole role;
  req<int> idx;
};
FF_VISITABLE_STRUCT(OpTensorSpec, role, idx);

OpTensorSpec input_tensor(int);
OpTensorSpec output_tensor(int);
OpTensorSpec weight_tensor(int);

using OpArgSpec = variant<ConcreteArgSpec,
                          IndexArgSpec,
                          OpArgRefSpec,
                          CheckedTypedFuture,
                          CheckedTypedFutureMap,
                          RuntimeArgRefSpec,
                          TaskInvocationSpec>;

struct OpTaskBinding {
  OpTaskBinding() = default;

  static_assert(is_subeq_variant<IndexTaskArgSpec, OpArgSpec>::value, "");

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
  std::unordered_map<slot_id, OpArgSpec> const &get_arg_bindings() const;

private:
  void insert_arg_spec(slot_id name, OpArgSpec const &arg_spec) {
    assert(!contains_key(this->arg_bindings, name));
    this->arg_bindings.insert({name, arg_spec});
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
  std::unordered_map<slot_id, OpArgSpec> arg_bindings;
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
OpTaskSignature get_op_signature(task_id_t const &);

/* std::unordered_map<int, OpTensorSpec> get_regions_idxs(TaskArgumentFormat
 * const &); */

/* TaskArgumentFormat compile_task_invocation(OpTaskSignature const &,
 * OpTaskBinding const &); */

} // namespace FlexFlow

#endif
