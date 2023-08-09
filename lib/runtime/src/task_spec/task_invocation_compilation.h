#ifndef _FLEXFLOW_RUNTIME_SRC_TASK_INVOCATION_COMPILATION_H
#define _FLEXFLOW_RUNTIME_SRC_TASK_INVOCATION_COMPILATION_H

#include "concrete_args_format.h"
#include "future_args_format.h"
#include "index_args_format.h"
#include "legion_backing.h"
#include "task_argument_accessor.h"
#include "task_invocation_args_format.h"
#include "tensor_args_format.h"

namespace FlexFlow {

using GenericTaskLauncher =
    variant<Legion::TaskLauncher, Legion::IndexTaskLauncher>;

TaskArgumentsFormat
    create_serializable_format(ConcreteArgsFormat const &,
                               FutureArgsFormat const &,
                               optional<TensorArgsFormat> const & = nullopt,
                               optional<IndexArgsFormat> const & = nullopt);

Legion::TaskArgument
    as_task_argument(ConcreteArgsFormat const &,
                     FutureArgsFormat const &,
                     optional<TensorArgsFormat> const & = nullopt,
                     optional<IndexArgsFormat> const & = nullopt);

Legion::ArgumentMap as_argument_map(IndexArgsFormat const &);
void add_futures(GenericTaskLauncher const &, FutureArgsFormat const &);

Legion::Future execute_task(Legion::TaskLauncher const &,
                            RuntimeBacking const &);
Legion::FutureMap execute_task(Legion::IndexTaskLauncher const &,
                               RuntimeBacking const &);

TaskReturnAccessor execute(ExecutableTaskInvocation const &,
                           ParallelComputationGraph const &,
                           RuntimeBacking const &,
                           EnableProfiling);
TaskReturnAccessor execute(TensorlessTaskInvocation const &,
                           TensorArgsFormat const &,
                           RuntimeBacking const &,
                           EnableProfiling);
TaskReturnAccessor execute(TensorlessTaskInvocation const &,
                           RuntimeBacking const &,
                           EnableProfiling);

} // namespace FlexFlow

#endif
