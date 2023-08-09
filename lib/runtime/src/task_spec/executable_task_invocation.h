#ifndef _FLEXFLOW_RUNTIME_SRC_EXECUTABLE_TASK_INVOCATION_H
#define _FLEXFLOW_RUNTIME_SRC_EXECUTABLE_TASK_INVOCATION_H

#include "runtime/task_spec/tensorless_task_invocation.h"
#include "task_invocation.h"

namespace FlexFlow {

using NonvariadicExecutableTensorSpec = parallel_tensor_guid_t;
using VariadicExecutableTensorSpec = std::vector<parallel_tensor_guid_t>;
using ExecutableTensorSpec =
    variant<NonvariadicExecutableTensorSpec, VariadicExecutableTensorSpec>;

bool is_variadic(ExecutableTensorSpec const &);
bool is_nonvariadic(ExecutableTensorSpec const &);
NonvariadicExecutableTensorSpec get_nonvariadic(ExecutableTensorSpec const &);
VariadicExecutableTensorSpec get_variadic(ExecutableTensorSpec const &);

struct ExecutableTaskBinding {
public:
  TensorlessTaskBinding tensorless;
  std::unordered_map<slot_id, ExecutableTensorSpec> tensor_bindings;
  optional<NonvariadicExecutableTensorSpec> domain_spec = nullopt;
};

bool is_variadic(ExecutableTaskBinding const &, slot_id);
bool is_nonvariadic(ExecutableTaskBinding const &, slot_id);

struct ExecutableTaskInvocation
    : public use_visitable_cmp<ExecutableTaskInvocation> {
public:
  ExecutableTaskInvocation() = delete;
  ExecutableTaskInvocation(task_id_t const &task_id,
                           ExecutableTaskInvocation const &binding);

public:
  task_id_t task_id;
  ExecutableTaskBinding binding;
};

} // namespace FlexFlow

#endif
