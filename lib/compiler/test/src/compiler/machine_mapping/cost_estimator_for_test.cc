#include "./cost_estimator_for_test.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/unmapped_op_cost_estimate_key.h"
#include "utils/containers/flatmap.h"
#include "utils/containers/map_keys.h"

namespace FlexFlow {

TestCostEstimator::TestCostEstimator(
  std::function<float(OpCostEstimateKey const &)> const &get_operator_cost,
  std::function<float(TensorSetMovement const &)> const & get_communication_cost)
  : get_operator_cost(get_operator_cost), 
    get_communication_cost(get_communication_cost)
{ }

float TestCostEstimator::estimate_cost(OpCostEstimateKey const &k) const {
  return this->get_operator_cost(k);
}

float TestCostEstimator::estimate_cost(TensorSetMovement const &m) const {
  return this->get_communication_cost(m);
}

CostEstimator make_fake_cost_estimator(
  std::function<float(OpCostEstimateKey const &)> const &get_operator_cost,
  std::function<float(TensorSetMovement const &)> const &get_communication_cost) {

  return CostEstimator::create<TestCostEstimator>(get_operator_cost, get_communication_cost);
}

CostEstimator make_fake_cost_estimator(
  std::unordered_map<OpCostEstimateKey, float> const &op_cost_map,
  std::unordered_map<TensorSetMovement, float> const &comm_cost_map) {
  return make_fake_cost_estimator(
    [op_cost_map](OpCostEstimateKey const &k) {
      return op_cost_map.at(k);
    },
    [comm_cost_map](TensorSetMovement const &m) {
      return  comm_cost_map.at(m);
    });
}

CostEstimator make_fake_cost_estimator(
  std::unordered_map<UnmappedOpCostEstimateKey, std::unordered_map<MachineView, float>> const &op_cost_map,
  std::unordered_map<TensorSetMovement, float> const &comm_cost_map) {

  auto de_nest_key = [](UnmappedOpCostEstimateKey const &k1, std::unordered_map<MachineView, float> const &v) 
    -> std::unordered_map<OpCostEstimateKey, float>
  {
    return map_keys(v, [&](MachineView const &k2) { return map_unmapped_op_cost_estimate_key(k1, k2); });
  };
  
  std::unordered_map<OpCostEstimateKey, float> mapped_costs = flatmap(op_cost_map, de_nest_key);

  return make_fake_cost_estimator(mapped_costs, comm_cost_map);
}

} // namespace FlexFlop
