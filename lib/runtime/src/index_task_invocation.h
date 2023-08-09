#ifndef _FLEXFLOW_RUNTIME_SRC_INDEX_TASK_INVOCAITON_H
#define _FLEXFLOW_RUNTIME_SRC_INDEX_TASK_INVOCATION_H

#include "task_invocation.h"

namespace FlexFlow {

using IndexArgSpec = variant<ConcreteArgSpec,
                             IndexArgSpec,
                             CheckedTypedFuture,
                             CheckedTypedFutureMap,
                             ArgRefSpec,
                             TaskInvocationSpec>;

struct IndexTaskBinding {
public:
  template <typename T>
  void bind_arg(slot_id name, IndexTypedTaskArg<T> const &);

  template <typename F,
            typename T = decltype(std::declval<F>()(
                std::declval<Legion::DomainPoint>()))>
  void bind_index_arg(slot_id name, F const &f) {
    this->insert_arg_spec(name, IndexArgSpec::create(f));
  }

public:
  TaskBinding standard_binding;
};

} // namespace FlexFlow

#endif
