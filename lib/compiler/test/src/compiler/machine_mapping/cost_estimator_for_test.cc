#include "./cost_estimator_for_test.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_tensor_set_movement.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/unmapped_op_cost_estimate_key.h"

namespace FlexFlow {

TestCostEstimator::TestCostEstimator(
    std::function<CostMetric(OpCostEstimateKey const &)> const
        &get_operator_cost,
    std::function<CostMetric(TensorSetMovement const &)> const
        &get_communication_cost)
    : get_operator_cost(get_operator_cost),
      get_communication_cost(get_communication_cost) {}

CostMetric TestCostEstimator::estimate_cost(OpCostEstimateKey const &k) const {
  return this->get_operator_cost(k);
}

CostMetric TestCostEstimator::estimate_cost(TensorSetMovement const &m) const {
  return this->get_communication_cost(m);
}

CostEstimator make_fake_cost_estimator(
    std::function<CostMetric(OpCostEstimateKey const &)> const
        &get_operator_cost,
    std::function<CostMetric(TensorSetMovement const &)> const
        &get_communication_cost) {

  return CostEstimator::create<TestCostEstimator>(get_operator_cost,
                                                  get_communication_cost);
}

CostEstimator make_fake_cost_estimator(
    std::unordered_map<OpCostEstimateKey, CostMetric> const &op_cost_map,
    std::unordered_map<TensorSetMovement, CostMetric> const &comm_cost_map) {
  return make_fake_cost_estimator(
      [op_cost_map](OpCostEstimateKey const &k) { return op_cost_map.at(k); },
      [comm_cost_map](TensorSetMovement const &m) {
        return comm_cost_map.at(m);
      });
}

} // namespace FlexFlow
