#ifndef _FLEXFLOW_EXECUTION_INCLUDE_LOCAL_EXECUTION_LOCAL_MODEL_TRAINING_INSTANCE_H
#define _FLEXFLOW_EXECUTION_INCLUDE_LOCAL_EXECUTION_LOCAL_MODEL_TRAINING_INSTANCE_H

#include "local_training_backing.h"
#include "op-attrs/ops/loss_functions.h"
#include "arg_backing.h"
#include "pcg/computation_graph.h"
#include "pcg/optimizer.h"
#include "pcg/tensor_guid_t.h"
#include "profiling.h"

namespace FlexFlow {

// struct TrainingConfig {
//   // TrainingConfig(ComputationGraph graph, Optimizer opt, EnableProfiling enable_profiling)
//   //   : computation_graph(graph), optimizer(opt), enable_profiling(enable_profiling) {};

//   ComputationGraph computation_graph;
//   Optimizer optimizer;
//   EnableProfiling enable_profiling;
// };
// FF_VISITABLE_STRUCT(TrainingConfig,
//                     computation_graph,
//                     optimizer,
//                     enable_profiling);

// struct TrainingComputationGraph {
//   ComputationGraph computation_graph;
//   tensor_guid_t logit_tensor;
//   tensor_guid_t label_tensor;
//   req<LossAttrs> loss;
//   // req<MetricsAttrs> metrics;
// };
// FF_VISITABLE_STRUCT(TrainingComputationGraph,
//                     computation_graph,
//                     logit_tensor,
//                     label_tensor,
//                     loss);

struct LocalModelTrainingInstance {
  LocalModelTrainingInstance() = delete;
  LocalModelTrainingInstance(
    ComputationGraph,
    Allocator,
    Optimizer,
    EnableProfiling,
    tensor_guid_t logit_tensor,
    tensor_guid_t label_tensor,
    LossAttrs,
    std::unordered_map<OperatorSlotBackingId, GenericTensorAccessorW &> slot_mapping,
    ArgBackingMapping);

  // TrainingConfig training_config;
  // TrainingComputationGraph training_computation_graph;
  ComputationGraph computation_graph;
  Optimizer optimizer;
  EnableProfiling enable_profiling;
  tensor_guid_t logit_tensor;
  tensor_guid_t label_tensor;
  LossAttrs loss;
  LocalTrainingBacking local_training_backing;

  void forward();
  void backward();
  void update();
  void reset_metrics();
};


} // namespace FlexFlow

#endif
