#ifndef _FLEXFLOW_TEST_COST_ESTIMATOR_H
#define _FLEXFLOW_TEST_COST_ESTIMATOR_H

#include "compiler/cost_estimator/cost_estimator.h"
#include "compiler/cost_estimator/op_cost_estimate_key.dtg.h"
#include "compiler/cost_estimator/tensor_set_movement.dtg.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_tensor_set_movement.dtg.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/unmapped_op_cost_estimate_key.dtg.h"
#include "compiler/machine_mapping/parallel_layer_guid_oblivious_machine_mapping.dtg.h"

namespace FlexFlow {

struct TestCostEstimator : public ICostEstimator {
  std::function<float(OpCostEstimateKey const &)> get_operator_cost;
  std::function<float(TensorSetMovement const &)> get_communication_cost;

  TestCostEstimator() = delete;
  TestCostEstimator(decltype(get_operator_cost) const &get_operator_cost,
                    decltype(get_communication_cost)
                        const &get_communication_cost);

  float estimate_cost(OpCostEstimateKey const &) const override;

  float estimate_cost(TensorSetMovement const &) const override;
};

CostEstimator make_fake_cost_estimator(
    std::function<float(OpCostEstimateKey const &)> const &get_operator_cost,
    std::function<float(TensorSetMovement const &)> const
        &get_communication_cost);

CostEstimator make_fake_cost_estimator(
    std::unordered_map<OpCostEstimateKey, float> const &op_cost_map,
    std::unordered_map<TensorSetMovement, float> const &comm_cost_map);

} // namespace FlexFlow

#endif
