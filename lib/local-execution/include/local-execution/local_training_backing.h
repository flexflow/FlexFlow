#ifndef _FLEXFLOW_LOCAL_EXECUTION_LOCAL_TRAINING_BACKING_H
#define _FLEXFLOW_LOCAL_EXECUTION_LOCAL_TRAINING_BACKING_H

#include "local-execution/local_slots_backing.h"
#include "local-execution/task_registry.h"

namespace FlexFlow {

enum class KernelType { FWD, BWD };

struct LocalTrainingBacking {
  LocalTrainingBacking(Allocator const &,
                       ComputationGraph const &,
                       TensorBackingMap const &,
                       RuntimeArgConfig const &);
  ~LocalTrainingBacking() = default;

  void execute_init();
  std::optional<float> execute_kernel(KernelType const &);
  void execute_update();

private:
  DeviceSpecific<DeviceStates>
      call_init_task_impl(task_id_t, TaskArgumentAccessor const &);
  std::optional<float> call_task_impl(task_id_t, TaskArgumentAccessor);

  TaskArgumentAccessor get_task_arg_accessor(OpTaskInvocation const &,
                                             layer_guid_t const &) const;

  Allocator allocator;
  ComputationGraph computation_graph;
  TaskRegistry task_registry;
  LocalSlotsBacking local_slots_backing;
};

std::vector<task_id_t> get_task_ids(ComputationGraphOpAttrs const &);

} // namespace FlexFlow

#endif
