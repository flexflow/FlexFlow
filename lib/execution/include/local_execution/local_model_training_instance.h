#ifndef _FLEXFLOW_EXECUTION_INCLUDE_LOCAL_EXECUTION_LOCAL_MODEL_TRAINING_INSTANCE_H
#define _FLEXFLOW_EXECUTION_INCLUDE_LOCAL_EXECUTION_LOCAL_MODEL_TRAINING_INSTANCE_H

#include "local_training_backing.h"
#include "metrics_functions.h"
#include "op-attrs/ops/loss_functions.h"
#include "pcg/computation_graph.h"
#include "pcg/optimizer.h"
#include "pcg/tensor_guid_t.h"
#include "profiling.h"

namespace FlexFlow {

struct LocalModelTrainingInstance {
  ComputationGraph computation_graph;
  Optimizer optimizer;
  EnableProfiling enable_profiling;
  tensor_guid_t logit_tensor;
  tensor_guid_t label_tensor;
  LossAttrs loss;
  MetricsAttrs metrics;
  LocalTrainingBacking local_training_backing;
};
FF_VISITABLE_STRUCT(LocalModelTrainingInstance,
                    computation_graph,
                    optimizer,
                    enable_profiling,
                    logit_tensor,
                    label_tensor,
                    loss,
                    metrics,
                    local_training_backing);

void initialize_backing(LocalModelTrainingInstance &,
                        std::unordered_map<OperatorSlotBackingId,
                                           GenericTensorAccessorW> slot_mapping,
                        size_t gpu_memory_size);
GenericTensorAccessorR forward(LocalModelTrainingInstance const &);
void backward(LocalModelTrainingInstance const &);
void update(LocalModelTrainingInstance const &);

} // namespace FlexFlow

#endif
