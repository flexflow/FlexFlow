#include "ffc/unity_algorithm.h"

using ::FlexFlow::utils::Node;
using ::FlexFlow::utils::Serial;
using ::FlexFlow::utils::Parallel;
using ::FlexFlow::utils::SerialParallelDecomposition;
using ::FlexFlow::opmeta::OperatorParameters;
using ::FlexFlow::opmeta::ParallelTensorShape;


namespace FlexFlow {
namespace ffc {

utils::SerialParallelDecomposition get_serial_parallel_decomposition(ParallelComputationGraph const &pcg) {
  return utils::get_serial_parallel_decomposition(unsafe_view_as_digraph(pcg.g));
}

/* ParallelComputationGraph get_subgraph(ParallelComputationGraph const &pcg, std::unordered_set<utils::Node> const &nodes) { */
/*   auto raw_subgraph = utils::get_subgraph<utils::AdjacencyMultiDiGraph>(pcg.g, nodes); */
/*   auto raw_nodeMap = utils::restrict_keys(pcg.nodeMap, nodes); */
/*   return { raw_subgraph, raw_nodeMap }; */
/* } */

struct GetNodes {
  template <typename T>
  std::unordered_set<utils::Node> operator()(T const &t) {
   return get_nodes(t); 
  }
};

std::unordered_set<utils::Node> get_nodes(utils::Serial const &serial) {
  return set_union(vector_transform(GetNodes{}, serial.children));
}

std::unordered_set<utils::Node> get_nodes(utils::Parallel const &parallel) {
  return set_union(vector_transform(GetNodes{}, parallel.children));
}

std::unordered_set<utils::Node> get_nodes(utils::Node const &node) {
  return {node};
}

std::unordered_set<utils::Node> get_nodes(utils::SerialParallelDecomposition const &sp) {
  return mpark::visit(GetNodes{}, sp);
}

float optimal_cost(ParallelComputationGraph const &g, std::unordered_set<MachineView> const &allowed_machine_views) {
  auto sp_decomposition = get_serial_parallel_decomposition(g);
  return optimal_cost(g, sp_decomposition, allowed_machine_views);
}

struct ICostEstimator {
  virtual float estimate_cost(OperatorParameters const &op, 
                              std::vector<ParallelTensorShape> const &inputs, 
                              MachineView const &mv) const = 0;
  virtual float estimate_cost(ParallelTensorShape const &tensor_shape,
                              MachineView const &src, 
                              MachineView const &dst);
};

struct OptimalCost {
  template <typename T>
  float operator()(T const &t) const {
    return this->optimal_cost(t);
  }

  float optimal_cost(Serial const &serial) const {
    return utils::sum(utils::vector_transform([&](mpark::variant<Parallel, Node> const &t) { return mpark::visit(*this, t); }, serial.children));
  }

  float optimal_cost(Parallel const &parallel) const {
  }

  float optimal_cost(Node const &node) const {
    // I need the input tensor shapes still and I need 
    for (MachineView const &machine_view : this->allowed_machine_views) {
    return this->cost_estimator.estimate_cost(pcg.get_op_params(node), );
  }
  
  ParallelComputationGraph const &pcg;
  std::unordered_set<MachineView> const &allowed_machine_views;
  ICostEstimator const &cost_estimator;
};

float optimal_cost(ParallelComputationGraph const &g, 
                   SerialParallelDecomposition const &sp_decomposition, 
                   std::unordered_set<MachineView> const &allowed_machine_views) {
  return mpark::visit(OptimalCost{g, allowed_machine_views}, sp_decomposition);
}

}
}
