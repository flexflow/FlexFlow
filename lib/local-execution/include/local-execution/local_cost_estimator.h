#ifndef _FLEXFLOW_LOCAL_EXECUTION_LOCAL_COST_ESTIMATOR_H
#define _FLEXFLOW_LOCAL_EXECUTION_LOCAL_COST_ESTIMATOR_H

#include "compiler/cost_estimate.h"

namespace FlexFlow {

struct LocalCostEstimator : public ICostEstimator {
  LocalCostEstimator() = default;
  LocalCostEstimator(LocalCostEstimator const &) = delete;
  LocalCostEstimator(LocalCostEstimator &&) = delete;
  ~LocalCostEstimator() = default;

  float estimate_cost(PCGOperatorAttrs const &op,
                      std::vector<ParallelTensorShape> const &inputs,
                      std::vector<ParallelTensorAttrs> const &weights,
                      std::vector<ParallelTensorAttrs> const &outputs,
                      MachineView const &mv) const override;

  float estimate_cost(ParallelTensorShape const &tensor_shape,
                      MachineView const &src,
                      MachineView const &dst) const override;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(LocalCostEstimator);

CostEstimator get_local_cost_estimator();

} // namespace FlexFlow

#endif
