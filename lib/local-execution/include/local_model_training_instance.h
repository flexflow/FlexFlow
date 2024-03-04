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

struct TrainingConfig {
  ComputationGraph computation_graph;
  Optimizer optimizer;
  req<EnableProfiling> enable_profiling;
};
FF_VISITABLE_STRUCT(TrainingConfig,
                    computation_graph,
                    optimizer,
                    enable_profiling);

struct TrainingComputationGraph {
  ComputationGraph const &computation_graph;
  tensor_guid_t logit_tensor;
  tensor_guid_t label_tensor;
  LossAttrs loss;
  req<MetricsAttrs> metrics;
};
FF_VISITABLE_STRUCT(TrainingComputationGraph,
                    computation_graph,
                    logit_tensor,
                    label_tensor,
                    loss,
                    metrics);

struct LocalModelTrainingInstance {
  TrainingConfig training_config;
  TrainingComputationGraph training_computation_graph;
  req<LocalTrainingBacking> local_training_backing;

  void forward();
  void backward();
  void update();
  void reset_metrics();
};
FF_VISITABLE_STRUCT(LocalModelTrainingInstance,
                    training_instance,
                    training_cg,
                    local_training_backing);

LocalModelTrainingInstance initialize_backing(
    ComputationGraph const &,
    Optimizer const &,
    EnableProfiling const,
    tensor_guid_t logit_tensor,
    tensor_guid_t label_tensor,
    LossAttrs loss,
    MetricsAttrs metrics,
    std::unordered_map<tensor_guid_t, GenericTensorAccessorW> const
        &slot_mapping);

} // namespace FlexFlow

#endif
