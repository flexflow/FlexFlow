#ifndef _FLEXFLOW_LOCAL_EXECUTION_LOCAL_BACKING_H
#define _FLEXFLOW_LOCAL_EXECUTION_LOCAL_BACKING_H

#include "task_registry.h"

namespace FlexFlow {

struct LocalTrainingBacking {
  LocalTrainingBacking(
      Allocator,
      ComputationGraph const &,
      TensorBackingMapping,
      RuntimeArgConfig);
  ~LocalTrainingBacking() = default;

  void execute_init();
  void execute_forward();
  void execute_backward();
  void execute_update();

  DeviceSpecific<DeviceStates> call_init_task_impl(task_id_t,
                                                   TaskArgumentAccessor);
  void call_task_impl(task_id_t, TaskArgumentAccessor);

  TaskArgumentAccessor get_task_arg_accessor(OpTaskInvocation const &, operator_guid_t const &);

private:
  Allocator allocator;
  ComputationGraph const & computation_graph;

  TaskRegistry task_registry;
};

// -- err (maybe): will this resolve correctly?
OpTaskInvocation init(CompGraphOperatorAttrs);
OpTaskInvocation forward(CompGraphOperatorAttrs);
OpTaskInvocation backward(CompGraphOperatorAttrs);

} // namespace FlexFlow

#endif
