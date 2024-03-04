#include "task_invocation_args_format.h"

namespace FlexFlow {

std::vector<ExecutableArgSpec>
    process_task_invocation_args(TensorlessTaskBinding const &binding,
                                 EnableProfiling enable_profiling,
                                 RuntimeBacking const &runtime_backing) {
  for (auto const &kv : get_args_of_type<TaskInvocationSpec>(binding)) {
    slot_id slot = kv.first;
    TaskInvocationSpec spec = kv.second;
    ExecutableTaskInvocation executable =
        resolve(spec.get_invocation(), enable_profiling, runtime_backing);

    TaskReturnAccessor ret_val = model.execute(executable);
  }
}

} // namespace FlexFlow
