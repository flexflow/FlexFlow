#ifndef _FLEXFLOW_TEST_COST_ESTIMATOR_H
#define _FLEXFLOW_TEST_COST_ESTIMATOR_H

#include "compiler/cost_estimator.h"
#include "compiler/op_cost_estimate_key.dtg.h"
#include "compiler/comm_cost_estimate_key.dtg.h"

namespace FlexFlow {

struct TestCostEstimator : public ICostEstimator {
  std::function<float(PCGOperatorAttrs const &, 
                      std::vector<ParallelTensorShape> const &,
                      std::vector<ParallelTensorShape> const &,
                      std::vector<ParallelTensorShape> const &,
                      MachineView const &)> get_operator_cost;
  std::function<float(ParallelTensorShape const &,
                      MachineView const &,
                      MachineView const &)> get_communication_cost;

  TestCostEstimator() = delete;
  TestCostEstimator(decltype(get_operator_cost) const &get_operator_cost,
                    decltype(get_communication_cost) const &get_communication_cost);

  float estimate_cost(PCGOperatorAttrs const &op,
                             std::vector<ParallelTensorShape> const &inputs,
                             std::vector<ParallelTensorShape> const &weights,
                             std::vector<ParallelTensorShape> const &outputs,
                             MachineView const &mv) const override;

  float estimate_cost(ParallelTensorShape const &tensor_shape,
                             MachineView const &src,
                             MachineView const &dst) const override;
};

CostEstimator make_cost_estimator(
  std::function<float(PCGOperatorAttrs const &, 
                      std::vector<ParallelTensorShape> const &,
                      std::vector<ParallelTensorShape> const &,
                      std::vector<ParallelTensorShape> const &,
                      MachineView const &)> const &get_operator_cost,
  std::function<float(ParallelTensorShape const &,
                      MachineView const &,
                      MachineView const &)> const &get_communication_cost);

CostEstimator make_cost_estimator(
  std::unordered_map<OpCostEstimateKey, float> const &,
  std::unordered_map<CommCostEstimateKey, float> const &);

} // namespace FlexFlow

#endif
