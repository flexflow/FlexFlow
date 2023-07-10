#ifndef _FLEXFLOW_RUNTIME_SRC_CONCRETE_ARGS_FORMAT_H
#define _FLEXFLOW_RUNTIME_SRC_CONCRETE_ARGS_FORMAT_H

#include "legion.h"
#include "runtime/task_spec/tensorless_task_invocation.h"
#include "task_argument_accessor.h"
#include "utils/stack_map.h"

namespace FlexFlow {

struct ConcreteArgsFormat {
public:
  ConcreteArgsFormat() = delete;
  ConcreteArgsFormat(
      Legion::Serializer const &sez,
      TaskArgumentsFormat *reserved_bytes_for_fmt,
      stack_map<slot_id, TaskArgumentFormat, MAX_NUM_TASK_ARGUMENTS> const
          &fmts)
      : sez(sez), fmts(fmts) {}

public:
  Legion::Serializer sez;
  TaskArgumentsFormat *reserved_bytes_for_fmt;
  stack_map<slot_id, TaskArgumentFormat, MAX_NUM_TASK_ARGUMENTS> fmts;
};

ConcreteArgsFormat
    process_concrete_args(std::unordered_map<slot_id, ConcreteArgSpec> const &);
ConcreteArgsFormat process_concrete_args(TensorlessTaskBinding const &);

} // namespace FlexFlow

#endif
