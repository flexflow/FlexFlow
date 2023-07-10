#ifndef _FLEXFLOW_RUNTIME_SRC_FUTURE_ARGS_FORMAT_H
#define _FLEXFLOW_RUNTIME_SRC_FUTURE_ARGS_FORMAT_H

#include "legion.h"
#include "task_argument_accessor.h"
#include "runtime/task_spec/tensorless_task_invocation.h"
#include "utils/stack_map.h"

namespace FlexFlow {

/**
 * \class FutureArgsFormat
 * \brief Structure for future arguments
 * 
 * Compiled from TensorlessIndexTaskBinding and TensorlessTaskBinding; Compiles to TaskArgumentsFormat and Legion::TaskArgument;
*/
struct FutureArgsFormat {
  std::vector<Legion::Future> futures;
  stack_map<slot_id, FutureArgumentFormat, MAX_NUM_TASK_ARGUMENTS> fmts;
};

/**
 * \fn FutureArgsFormat process_future_args(TensorlessTaskBinding const &)
 * \brief Processes future arguments
 * \param binding A TensorlessTaskBinding to pass into get_args_of_type<T>()
 * 
 * \\todo add more detailed description
*/
FutureArgsFormat process_future_args(TensorlessTaskBinding const &binding);

/**
 * \fn CheckedTypedFuture resolve_future_map_arg(CheckedTypedFuture const &, Legion::Domain const &)
 * \param checkedtypedfuture CheckedTypedFuture
 * \param domain Legion::Domain
 * \\todo doesn't have definition
*/
CheckedTypedFuture resolve_future_map_arg(CheckedTypedFuture const &checkedtypedfuture,
                                          Legion::Domain const &domain);

} // namespace FlexFlow

#endif
