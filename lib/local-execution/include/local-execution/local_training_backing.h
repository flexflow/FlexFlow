#ifndef _FLEXFLOW_LOCAL_EXECUTION_LOCAL_TRAINING_BACKING_H
#define _FLEXFLOW_LOCAL_EXECUTION_LOCAL_TRAINING_BACKING_H

#include "local-execution/local_slots_backing.h"
#include "local-execution/task_registry.h"
#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"
#include "pcg/computation_graph.dtg.h"
#include "pcg/optimizer_attrs.dtg.h"

namespace FlexFlow {

using PerLayerElapsedTime =
    std::unordered_map<layer_guid_t, std::optional<float>>;

struct LocalTrainingBacking {
  LocalTrainingBacking(Allocator const &,
                       ComputationGraph const &,
                       TensorBackingMap const &,
                       RuntimeArgConfig const &);
  void register_and_allocate_layer(layer_guid_t const &);
  void allocate_layer_optimizer_tensors(layer_guid_t const &,
                                        OptimizerAttrs const &);

  void execute_init(layer_guid_t const &);
  std::optional<float> execute_forward(layer_guid_t const &);
  void compute_loss(LossAttrs const &loss_attrs,
                    tensor_guid_t const &logit_tensor,
                    tensor_guid_t const &label_tensor);
  std::optional<float> execute_backward(layer_guid_t const &);
  void execute_update(layer_guid_t const &, OptimizerAttrs const &);

  TaskArgumentAccessor get_task_arg_accessor(TaskInvocation const &) const;
  TaskArgumentAccessor get_op_task_arg_accessor(OpTaskInvocation const &,
                                                layer_guid_t const &) const;

  void insert_tensor(tensor_guid_t const &, GenericTensorAccessorW const &);

private:
  DeviceSpecificDeviceStates call_init_task_impl(task_id_t,
                                                 TaskArgumentAccessor const &);
  std::optional<float> call_task_impl(task_id_t, TaskArgumentAccessor);

private:
  Allocator allocator;
  ComputationGraph computation_graph;
  TaskRegistry task_registry;
  LocalSlotsBacking local_slots_backing;
};

} // namespace FlexFlow

#endif
