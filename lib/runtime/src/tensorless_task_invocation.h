#ifndef _FLEXFLOW_RUNTIME_SRC_TENSORLESS_TASK_INVOCATION_H
#define _FLEXFLOW_RUNTIME_SRC_TENSORLESS_TASK_INVOCATION_H

#include "task_invocation.h"
#include "utils/visitable.h"

namespace FlexFlow {

using StandardExecutableArgSpec = variant<ConcreteArgSpec, CheckedTypedFuture, CheckedTypedFutureMap, TaskInvocationSpec, IndexTaskInvocationSpec>;
using IndexExecutableArgSpec = variant<IndexArgSpec, IndexTaskInvocationSpec>;

struct TensorlessTaskBinding : public use_visitable_cmp<TensorlessTaskBinding> {
public:
  TensorlessTaskBinding() = default;

  using ArgType = StandardExecutableArgSpec;

  template <typename T> void bind_arg(slot_id name, T const &);
  template <typename T> void bind_arg(slot_id name, TypedFuture<T> const &);
  template <typename T> void bind_arg(slot_id name, TypedFutureMap<T> const &);
  template <typename T> void bind_arg(TypedTaskInvocation<T> const &);
public:
  std::unordered_map<slot_id, StandardExecutableArgSpec> arg_bindings;
};
static_assert(is_well_behaved_value_type<TensorlessTaskBinding>::value, "");

struct TensorlessIndexTaskBinding : public use_visitable_cmp<TensorlessIndexTaskBinding> {
  TensorlessIndexTaskBinding() = delete;
  TensorlessIndexTaskBinding(MachineView const &);

  using ArgType = variant_join<StandardExecutableArgSpec, IndexExecutableArgSpec>;

  template <typename T> void bind_arg(slot_id name, T const &);
  template <typename T> void bind_arg(slot_id name, TypedFuture<T> const &);
  template <typename T> void bind_arg(slot_id name, TypedFutureMap<T> const &);
  template <typename T> void bind_arg(TypedTaskInvocation<T> const &);

  template <typename F, typename T = decltype(std::declval<F>()(std::declval<Legion::DomainPoint>()))>
  void bind_index_arg(slot_id name, F const &f);
public:
  TensorlessTaskBinding standard;
  std::unordered_map<slot_id, IndexExecutableArgSpec> arg_bindings;
};
static_assert(is_well_behaved_value_type<TensorlessIndexTaskBinding>::value, "");

template <typename BindingType, typename T>
std::unordered_map<slot_id, T> get_args_of_type(BindingType const &binding) {
  using ArgType = typename BindingType::ArgType;
  static_assert(is_in_variant<T, ArgType>::value, "");
  return map_values(filter_values(binding.arg_bindings, 
                                  [](ArgType const &s) { return holds_alternative<T>(s); }),
                    [](ArgType const &s) { return get<T>(s); });
}

template <typename T>
std::unordered_map<slot_id, T> get_args_of_type(TensorlessTaskBinding const &binding) {
  return get_args_of_type<TensorlessTaskBinding, T>(binding);
}

template <typename T>
std::unordered_map<slot_id, T> get_args_of_type(TensorlessIndexTaskBinding const &binding) {
  return get_args_of_type<TensorlessIndexTaskBinding, T>(binding);
}

struct TensorlessTaskInvocation : public use_visitable_cmp<TensorlessTaskInvocation> {
public:
  TensorlessTaskInvocation() = delete;
  TensorlessTaskInvocation(task_id_t const &task_id, TensorlessTaskBinding const &binding); 

public:
  task_id_t task_id;
  TensorlessTaskBinding binding;
};
static_assert(is_well_behaved_value_type<TensorlessTaskInvocation>::value, "");

struct TensorlessIndexTaskInvocation : public use_visitable_cmp<TensorlessIndexTaskInvocation> {
public:
  TensorlessIndexTaskInvocation() = delete;
  TensorlessIndexTaskInvocation(task_id_t const &task_id, TensorlessIndexTaskBinding const &binding); 

public:
  task_id_t task_id;
  TensorlessIndexTaskBinding binding;
};
static_assert(is_well_behaved_value_type<TensorlessIndexTaskInvocation>::value, "");


}

#endif
