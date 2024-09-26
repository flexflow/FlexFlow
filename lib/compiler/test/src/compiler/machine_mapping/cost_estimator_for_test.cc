#include "./cost_estimator_for_test.h"

namespace FlexFlow {

TestCostEstimator::TestCostEstimator(
  std::function<float(PCGOperatorAttrs const &, 
                      std::vector<ParallelTensorShape> const &,
                      std::vector<ParallelTensorShape> const &,
                      std::vector<ParallelTensorShape> const &,
                      MachineView const &)> const &get_operator_cost,
  std::function<float(ParallelTensorShape const &,
                      MachineView const &,
                      MachineView const &)> const &get_communication_cost)
  : get_operator_cost(get_operator_cost), 
    get_communication_cost(get_communication_cost)
{ }

float TestCostEstimator::estimate_cost(PCGOperatorAttrs const &op,
                           std::vector<ParallelTensorShape> const &inputs,
                           std::vector<ParallelTensorShape> const &weights,
                           std::vector<ParallelTensorShape> const &outputs,
                           MachineView const &mv) const {
  return this->get_operator_cost(op, inputs, weights, outputs, mv);
}

float TestCostEstimator::estimate_cost(ParallelTensorShape const &tensor_shape,
                                       MachineView const &src,
                                       MachineView const &dst) const {
  return this->get_communication_cost(tensor_shape, src, dst);
}

CostEstimator make_cost_estimator(
  std::function<float(PCGOperatorAttrs const &, 
                      std::vector<ParallelTensorShape> const &,
                      std::vector<ParallelTensorShape> const &,
                      std::vector<ParallelTensorShape> const &,
                      MachineView const &)> const &get_operator_cost,
  std::function<float(ParallelTensorShape const &,
                      MachineView const &,
                      MachineView const &)> const &get_communication_cost) {
  return CostEstimator::create<TestCostEstimator>(get_operator_cost, get_communication_cost);
}

CostEstimator make_cost_estimator(
  std::unordered_map<OpCostEstimateKey, float> const &op_cost_map,
  std::unordered_map<CommCostEstimateKey, float> const &comm_cost_map) {
  return make_cost_estimator(
    [op_cost_map](PCGOperatorAttrs const &op_attrs,
       std::vector<ParallelTensorShape> const &input_shapes,
       std::vector<ParallelTensorShape> const &weight_shapes,
       std::vector<ParallelTensorShape> const &output_shapes,
       MachineView const &machine_view) {
      
      OpCostEstimateKey key = OpCostEstimateKey{
        /*op_attrs=*/op_attrs,
        /*input_shapes=*/input_shapes,
        /*weight_shapes=*/weight_shapes,
        /*output_shapes=*/output_shapes,
        /*machine_view=*/machine_view,
      };

      return op_cost_map.at(key);
    },
    [comm_cost_map](ParallelTensorShape const &parallel_tensor_shape,
                    MachineView const &src_machine_view,
                    MachineView const &dst_machine_view) {
      
      CommCostEstimateKey key = CommCostEstimateKey{
        /*parallel_tensor_shape=*/parallel_tensor_shape,
        /*src_machine_view=*/src_machine_view,
        /*dst_machine_view=*/dst_machine_view,
      };
      
      return comm_cost_map.at(key);
    });
}

} // namespace FlexFlow
