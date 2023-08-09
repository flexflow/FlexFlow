#include "training_pcg.h"
#include "loss_functions.h"

namespace FlexFlow {

std::vector<TaskInvocation> init_operators(TrainingPCG const &training) {
  std::unordered_map<operator_guid_t, OpTaskInvocation> init_invocations =
      init(training.pcg);
  return values(resolve(training.pcg, init_invocations));
}

std::vector<TaskInvocation> forward(TrainingPCG const &training) {
  std::unordered_map<operator_guid_t, OpTaskInvocation> forward_invocations =
      forward(training.pcg);
  return values(resolve(training.pcg, forward_invocations));
}

TaskInvocation compute_metrics(TrainingPCG const &training,
                               PerfMetrics const &all_metrics) {
  return compute_and_update_metrics(training.metrics,
                                    all_metrics,
                                    training.logit_tensor,
                                    training.label_tensor);
}

std::vector<TaskInvocation> backward(TrainingPCG const &training,
                                     PerfMetrics const &all_metrics) {
  TaskInvocation compute_metrics_invoke =
      compute_metrics(training, all_metrics);
  TaskInvocation loss_invoke =
      backward(training.loss, training.logit_tensor, training.label_tensor);
  std::vector<TaskInvocation> backward_pass =
      values(resolve(training.pcg, backward(training.pcg)));

  std::vector<TaskInvocation> result;
  result.push_back(compute_metrics_invoke);
  result.push_back(loss_invoke);
  extend(result, backward_pass);
  return result;
}

} // namespace FlexFlow
