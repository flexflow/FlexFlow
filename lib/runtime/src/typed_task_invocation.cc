#include "typed_task_invocation.h"

namespace FlexFlow {

TaskInvocationSpec::TaskInvocationSpec(std::type_index const &type_idx,
                                       TaskInvocation const &invocation)
    : type_idx(type_idx) {
  this->invocation = std::make_shared<TaskInvocation>(invocation);
}

} // namespace FlexFlow
