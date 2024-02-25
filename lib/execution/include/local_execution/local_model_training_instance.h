#ifndef _FLEXFLOW_RUNTIME_INCLUDE_RUNTIME_LOCAL_MODEL_TRAINING_INSTANCE_H
#define _FLEXFLOW_RUNTIME_INCLUDE_RUNTIME_LOCAL_MODEL_TRAINING_INSTANCE_H

#include "pcg/computation_graph.h"
#include "pcg/optimizer.h"
#include "pcg/tensor_guid_t.h"
#include "profiling.h"
#include "metrics_functions.h"
#include "op-attrs/ops/loss_functions.h"
#include "local_training_backing.h"

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

void initialize_backing(LocalModelTrainingInstance &);
GenericTensorAccessorR forward(LocalModelTrainingInstance const &);
void backward(LocalModelTrainingInstance const &);
void update(LocalModelTrainingInstance const &);

} // namespace FlexFlow

#endif
