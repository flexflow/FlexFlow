#ifndef _FLEXFLOW_RUNTIME_SRC_TENSORLESS_TASK_INVOCATION_H
#define _FLEXFLOW_RUNTIME_SRC_TENSORLESS_TASK_INVOCATION_H

#include "concrete_arg.h"
#include "index_arg.h"
#include "pcg/machine_view.h"
#include "slot_id.h"
#include "typed_future.h"
#include "typed_future_map.h"
#include "typed_task_invocation.h"
#include "utils/visitable.h"

namespace FlexFlow {

using StandardExecutableArgSpec = variant<ConcreteArgSpec,
                                          CheckedTypedFuture,
                                          CheckedTypedFutureMap,
                                          TaskInvocationSpec,
                                          IndexTaskInvocationSpec>;
using IndexExecutableArgSpec = variant<IndexArgSpec, IndexTaskInvocationSpec>;

struct TensorlessTaskBinding : public use_visitable_cmp<TensorlessTaskBinding> {
public:
  TensorlessTaskBinding() = default;

  using ArgType = StandardExecutableArgSpec;

  template <typename T>
  void bind_arg(slot_id name, T const &);
  template <typename T>
  void bind_arg(slot_id name, TypedFuture<T> const &);
  template <typename T>
  void bind_arg(slot_id name, TypedFutureMap<T> const &);
  template <typename T>
  void bind_arg(TypedStandardTaskInvocation<T> const &);

public:
  std::unordered_map<slot_id, StandardExecutableArgSpec> arg_bindings;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_HASH(TensorlessTaskBinding);

struct TensorlessIndexTaskBinding
    : public use_visitable_cmp<TensorlessIndexTaskBinding> {
  TensorlessIndexTaskBinding() = delete;
  TensorlessIndexTaskBinding(MachineView const &);

  using ArgType =
      variant_join<StandardExecutableArgSpec, IndexExecutableArgSpec>;

  template <typename T>
  void bind_arg(slot_id name, T const &);
  template <typename T>
  void bind_arg(slot_id name, TypedFuture<T> const &);
  template <typename T>
  void bind_arg(slot_id name, TypedFutureMap<T> const &);
  template <typename T>
  void bind_arg(TypedStandardTaskInvocation<T> const &);

  template <typename F,
            typename T = decltype(std::declval<F>()(
                std::declval<Legion::DomainPoint>()))>
  void bind_index_arg(slot_id name, F const &f);

public:
  TensorlessTaskBinding standard;
  std::unordered_map<slot_id, IndexExecutableArgSpec> arg_bindings;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_HASH(TensorlessIndexTaskBinding);

template <typename BindingType, typename T>
std::unordered_map<slot_id, T> get_args_of_type(BindingType const &binding) {
  using ArgType = typename BindingType::ArgType;
  static_assert(is_in_variant<T, ArgType>::value, "");
  return map_values(
      filter_values(binding.arg_bindings,
                    [](ArgType const &s) { return holds_alternative<T>(s); }),
      [](ArgType const &s) { return get<T>(s); });
}

template <typename T>
std::unordered_map<slot_id, T>
    get_args_of_type(TensorlessTaskBinding const &binding) {
  return get_args_of_type<TensorlessTaskBinding, T>(binding);
}

template <typename T>
std::unordered_map<slot_id, T>
    get_args_of_type(TensorlessIndexTaskBinding const &binding) {
  return get_args_of_type<TensorlessIndexTaskBinding, T>(binding);
}

struct TensorlessTaskInvocation {
  req<task_id_t> task_id;
  req<TensorlessTaskBinding> binding;
};
FF_VISITABLE_STRUCT(TensorlessTaskInvocation, task_id, binding);

struct TensorlessIndexTaskInvocation {
  req<task_id_t> task_id;
  req<TensorlessIndexTaskBinding> binding;
};
FF_VISITABLE_STRUCT(TensorlessIndexTaskInvocation, task_id, binding);

} // namespace FlexFlow

#endif
