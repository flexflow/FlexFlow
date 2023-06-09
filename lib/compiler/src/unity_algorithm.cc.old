#include "ffc/unity_algorithm.h"

namespace FlexFlow {
namespace ffc {

SerialParallelDecomposition get_serial_parallel_decomposition(ParallelComputationGraph const &pcg) {
  return get_serial_parallel_decomposition(unsafe_view_as_digraph(pcg.g));
}

std::vector<MultiDiEdge> get_sorted_node_input_edges(ParallelComputationGraph const &pcg, Node const &n) {
  std::unordered_map<std::size_t, std::unordered_set<MultiDiEdge>> incoming_edges = get_incoming_edges_by_idx(pcg.g, n);

  std::vector<MultiDiEdge> result;
  for (std::size_t i = 0; i < incoming_edges.size(); i++) {
    result.push_back(get_only(incoming_edges.at(i)));
  }

  return result;
}

std::unordered_map<MultiDiEdge, opmeta::ParallelTensorShape> infer_tensor_shapes(ParallelComputationGraph const &pcg) {
  std::unordered_map<MultiDiEdge, opmeta::ParallelTensorShape> result;
  for (Node const &n : get_topological_ordering(pcg.g)) {
    PCGOperatorAttrs op = pcg.at(n); 

    std::vector<ParallelTensorShape> input_tensor_shapes = vector_transform(
        [&](MultiDiEdge const &e) { return result.at(e); }, 
        get_sorted_node_input_edges(pcg, n)
    );

    std::vector<ParallelTensorShape> output_tensor_shapes = get_output_shapes(op, input_tensor_shapes);
  
    auto outgoing_edges = get_outgoing_edges_by_idx(pcg.g, n);

    for (std::size_t i = 0; i < output_tensor_shapes.size(); i++) {
      if (contains_key(outgoing_edges, i)) {
        for (MultiDiEdge const &e : outgoing_edges.at(i)) {
          result.insert({ e, output_tensor_shapes[i] });
        }
      }
    }
  }

  assert (result.size() == get_edges(pcg.g).size());

  return result;
}

/* ParallelComputationGraph get_subgraph(ParallelComputationGraph const &pcg, std::unordered_set<Node> const &nodes) { */
/*   auto raw_subgraph = get_subgraph<AdjacencyMultiDiGraph>(pcg.g, nodes); */
/*   auto raw_nodeMap = restrict_keys(pcg.nodeMap, nodes); */
/*   return { raw_subgraph, raw_nodeMap }; */
/* } */

/* struct GetNodes { */
/*   template <typename T> */
/*   std::unordered_set<Node> operator()(T const &t) { */
/*    return get_nodes(t); */ 
/*   } */
/* }; */

/* std::unordered_set<Node> get_nodes(Serial const &serial) { */
/*   return set_union(vector_transform(GetNodes{}, serial.children)); */
/* } */

/* std::unordered_set<Node> get_nodes(Parallel const &parallel) { */
/*   return set_union(vector_transform(GetNodes{}, parallel.children)); */
/* } */

/* std::unordered_set<Node> get_nodes(Node const &node) { */
/*   return {node}; */
/* } */

/* std::unordered_set<Node> get_nodes(SerialParallelDecomposition const &sp) { */
/*   return mpark::visit(GetNodes{}, sp); */
/* } */

/* float optimal_cost(ParallelComputationGraph const &g, std::unordered_set<MachineView> const &allowed_machine_views) { */
/*   auto sp_decomposition = get_serial_parallel_decomposition(g); */
/*   return optimal_cost(g, sp_decomposition, allowed_machine_views); */
/* } */

struct OpenSubParallelComputationGraph {
  std::unique_ptr<IDownwardOpenMultiDiGraphView const> g;
  std::unordered_map<Node, opmeta::OperatorParameters> nodeMap;
  std::unordered_map<DownwardOpenMultiDiEdge, MachineView> const &output_machine_views;
};

using SubParallelComputationGraph = mpark::variant<
  ParallelComputationGraph,
  OpenSubParallelComputationGraph
>;

struct ICostEstimator {
  virtual float estimate_cost(OperatorParameters const &op, 
                              std::vector<ParallelTensorShape> const &inputs, 
                              MachineView const &mv) const = 0;
  virtual float estimate_cost(ParallelTensorShape const &tensor_shape,
                              MachineView const &src, 
                              MachineView const &dst) = 0;
};

std::size_t num_nodes(OpenSubParallelComputationGraph const &g) {
  return num_nodes(*g.g);
}

std::size_t num_nodes(ParallelComputationGraph const &g) {
  return num_nodes(g.g);
}

bool is_base_case(SubParallelComputationGraph const &g) {
  if (mpark::holds_alternative<OpenSubParallelComputationGraph>(g)) {
    return num_nodes(mpark::get<OpenSubParallelComputationGraph>(g)) == 1;
  } else {
    return num_nodes(mpark::get<ParallelComputationGraph>(g)) == 2;
  }
}

std::pair<
  OpenSubParallelComputationGraph, 
  OpenSubParallelComputationGraph,
> apply_split(OpenSubParallelComputationGraph const &g, GraphSplit const &split) {
  
}

std::pair<
  OpenSubParallelComputationGraph,
  ParallelComputationGraph
> apply_split(ParallelComputationGraph const &g, GraphSplit const &split) {

}

float base_case(OpenSubParallelComputationGraph const &g);
float base_case(ParallelComputationGraph const &g);

float internal_optimal_cost(SubParallelComputationGraph const &g, 
                            ICostEstimator const &cost_estimator, 
                            SerialParallelDecomposition const &sp_decomposition,
                            std::function<std::unordered_set<MachineView>(opmeta::OperatorParameters const &, Resources const &)> const &f
                            ) {
  if (is_base_case(g)) {
    // base case
  } else {
    // non-base-case 
  }
}

struct OptimalCost {
  template <typename T>
  float operator()(T const &t) const {
    return this->optimal_cost(t);
  }

  float optimal_cost(Serial const &serial) const {
    return sum(vector_transform([&](mpark::variant<Parallel, Node> const &t) { return mpark::visit(*this, t); }, serial.children));
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
