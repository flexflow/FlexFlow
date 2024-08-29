#ifndef _FLEXFLOW_TEST_COST_ESTIMATOR_H
#define _FLEXFLOW_TEST_COST_ESTIMATOR_H

#include "compiler/cost_estimator.h"

namespace FlexFlow {

struct CostEstimatorForTest : public ICostEstimator {
  inline float estimate_cost(PCGOperatorAttrs const &op,
                      std::vector<ParallelTensorShape> const &inputs,
                      std::vector<ParallelTensorAttrs> const &weights,
                      std::vector<ParallelTensorAttrs> const &outputs,
                      MachineView const &mv) const override {
    return 1;
  }
  inline float estimate_cost(ParallelTensorShape const &tensor_shape,
                      MachineView const &src,
                      MachineView const &dst) const override {
    return 1;
  }
};

} // namespace FlexFlow

#endif
