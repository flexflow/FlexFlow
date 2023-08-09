#ifndef _FLEXFLOW_RUNTIME_SRC_TRAINING_PCG_H
#define _FLEXFLOW_RUNTIME_SRC_TRAINING_PCG_H

#include "metrics_functions.h"
#include "op-attrs/ops/loss_functions.h"
#include "parallel_computation_graph.h"

namespace FlexFlow {

struct TrainingPCG {
public:
  ParallelComputationGraph pcg;
  parallel_tensor_guid_t logit_tensor;
  parallel_tensor_guid_t label_tensor;
  LossAttrs loss;
  MetricsAttrs metrics;
};

std::vector<TaskInvocation> init_operators(TrainingPCG const &);
std::vector<TaskInvocation> forward(TrainingPCG const &);
std::vector<TaskInvocation> backward(TrainingPCG const &);
TaskInvocation compute_metrics(TrainingPCG const &, PerfMetrics const &);

} // namespace FlexFlow

#endif
