#ifndef _FLEXFLOW_RUNTIME_SRC_TENSORLESS_TASK_INVOCATION_H
#define _FLEXFLOW_RUNTIME_SRC_TENSORLESS_TASK_INVOCATION_H

#include "task_invocation.h"
#include "utils/visitable.h"

namespace FlexFlow {

using ExecutableArgSpec = variant<ConcreteArgSpec,
                                  IndexArgSpec,
                                  CheckedTypedFuture,
                                  CheckedTypedFutureMap,
                                  TaskInvocationSpec>;

struct TensorlessTaskBinding : public use_visitable_cmp<TensorlessTaskBinding> {
public:
  static TensorlessTaskBinding index_launch(MachineView const &);
  static TaskBinding standard_launch();

  template <typename T>
  void bind_arg(slot_id name, T const &);
  template <typename T>
  void bind_arg(slot_id name, TypedFuture<T> const &);
  template <typename T>
  void bind_arg(slot_id name, TypedFutureMap<T> const &);
  template <typename T>
  void bind_arg(TypedTaskInvocation<T> const &);

  template <typename F,
            typename T = decltype(std::declval<F>()(
                std::declval<Legion::DomainPoint>()))>
  void bind_index_arg(slot_id name, F const &f);

public:
  InvocationType invocation_type;
  std::unordered_map<slot_id, ExecutableArgSpec> arg_bindings;
  optional<MachineView> domain_view = nullopt;
};

template <typename T>
std::unordered_map<slot_id, T>
    get_args_of_type(TensorlessTaskBinding const &binding) {
  static_assert(is_in_variant<T, ExecutableArgSpec>::value, "");
  return map_values(filter_values(binding.arg_bindings,
                                  [](ExecutableArgSpec const &s) {
                                    return holds_alternative<T>(s);
                                  }),
                    [](ExecutableArgSpec const &s) { return get<T>(s); });
}

struct TensorlessTaskInvocation
    : public use_visitable_cmp<TensorlessTaskInvocation> {
public:
  TensorlessTaskInvocation() = delete;
  TensorlessTaskInvocation(task_id_t const &task_id,
                           TensorlessTaskBinding const &binding);

public:
  task_id_t task_id;
  TensorlessTaskBinding binding;
};

} // namespace FlexFlow

#endif
