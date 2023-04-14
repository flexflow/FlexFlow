#include "compiler/unity_algorithm.h"
#include "utils/graph/algorithms.h"

namespace FlexFlow {

struct ParallelComputationGraph {
  IMultiDiGraphView const &graph() const;
  PCGOperatorAttrs const &at(Node const &) const;
};

SerialParallelDecomposition get_serial_parallel_decomposition(ParallelComputationGraph const &pcg) {
  return get_serial_parallel_decomposition(*unsafe_view_as_digraph(pcg.graph()));
}

std::vector<MultiDiEdge> get_sorted_node_input_edges(ParallelComputationGraph const &pcg, Node const &n) {
  std::unordered_map<std::size_t, std::unordered_set<MultiDiEdge>> incoming_edges = get_incoming_edges_by_idx(pcg.graph(), n);

  std::vector<MultiDiEdge> result;
  for (std::size_t i = 0; i < incoming_edges.size(); i++) {
    result.push_back(get_only(incoming_edges.at(i)));
  }

  return result;
}

std::unordered_map<MultiDiEdge, ParallelTensorShape> infer_tensor_shapes(ParallelComputationGraph const &pcg) {
  std::unordered_map<MultiDiEdge, ParallelTensorShape> result;
  for (Node const &n : get_topological_ordering(pcg.graph())) {
    PCGOperatorAttrs op = pcg.at(n); 

    std::vector<ParallelTensorShape> input_tensor_shapes = vector_transform(
        [&](MultiDiEdge const &e) { return result.at(e); }, 
        get_sorted_node_input_edges(pcg, n)
    );

    std::vector<ParallelTensorShape> output_tensor_shapes = get_output_shapes(op, input_tensor_shapes);
  
    auto outgoing_edges = get_outgoing_edges_by_idx(pcg.graph(), n);

    for (std::size_t i = 0; i < output_tensor_shapes.size(); i++) {
      if (contains_key(outgoing_edges, i)) {
        for (MultiDiEdge const &e : outgoing_edges.at(i)) {
          result.insert({ e, output_tensor_shapes[i] });
        }
      }
    }
  }

  assert (result.size() == get_edges(pcg.graph()).size());

  return result;
}

/* ParallelComputationGraph get_subgraph(ParallelComputationGraph const &pcg, std::unordered_set<Node> const &nodes) { */
/*   auto raw_subgraph = get_subgraph<AdjacencyMultiDiGraph>(pcg.g, nodes); */
/*   auto raw_nodeMap = restrict_keys(pcg.nodeMap, nodes); */
/*   return { raw_subgraph, raw_nodeMap }; */
/* } */

struct GetNodes {
  template <typename T>
  std::unordered_set<Node> operator()(T const &t) {
   return get_nodes(t);
  }
};

std::unordered_set<Node> get_nodes(SerialParallelDecomposition const &sp) {
   return visit(GetNodes{}, sp); 
} 

std::unordered_set<Node> get_nodes(Serial const &serial) {
  return set_union(vector_transform([](variant<Parallel, Node> const child) { return visit(GetNodes{}, child);}, serial.children));
}

std::unordered_set<Node> get_nodes(Parallel const &parallel) {
  return set_union(vector_transform([](variant<Serial, Node> const child) { return visit(GetNodes{}, child);}, parallel.children));
}

std::unordered_set<Node> get_nodes(Node const &node) {
  return {node};
}

/* float optimal_cost(ParallelComputationGraph const &g, std::unordered_set<MachineView> const &allowed_machine_views) { */
/*   auto sp_decomposition = get_serial_parallel_decomposition(g); */
/*   return optimal_cost(g, sp_decomposition, allowed_machine_views); */
/* } */

// struct OpenSubParallelComputationGraph {
//   std::unique_ptr<IDownwardOpenMultiDiGraphView const> g;
//   std::unordered_map<Node, PCGOperatorAttrs> nodeMap;
//   std::unordered_map<DownwardOpenMultiDiEdge, MachineView> const &output_machine_views;
// };

// Move them to the proper place after finalizing the design
template<class T>
void minimize(T &t, T const &v) {
  t = std::min(t, v);
}

using SubParallelComputationGraph = LabelledOpenMultiDiGraph<PCGOperatorAttrs, ParallelTensorShape, MachineView>;

std::unordered_set<Node> get_closed_sources(IOpenMultiDiGraphView const &g);
std::unordered_set<Node> get_closed_sinks(IOpenMultiDiGraphView const &g);
std::unordered_set<Node> get_open_sources(IOpenMultiDiGraphView const &g);
std::unordered_set<Node> get_open_sinks(IOpenMultiDiGraphView const &g);

std::unordered_set<MultiDiEdge> get_cut(IOpenMultiDiGraphView const &g, GraphSplit const &split);

enum class InputSettings {
  INCLUDE_INPUTS,
  EXCLUDE_INPUTS
};

enum class OutputSettings {
  INCLUDE_OUTPUTS,
  EXCLUDE_OUTPUTS
};

SubParallelComputationGraph get_subgraph(
  SubParallelComputationGraph const &g,
  std::unordered_set<Node> const &nodes,
  InputSettings input_settings,
  OutputSettings output_settings) {
  
  auto base_graph = g.graph(); // Need the access to the base graph
  auto base_subgraph = get_subgraph(base_graph, nodes);

  // Need interfaces to create a labelled open graph
}

struct ICostEstimator {
  virtual float estimate_cost(PCGOperatorAttrs const &op, 
                              std::vector<ParallelTensorShape> const &inputs, 
                              MachineView const &mv) const = 0;
  virtual float estimate_cost(ParallelTensorShape const &tensor_shape,
                              MachineView const &src, 
                              MachineView const &dst) = 0;
};

float estimate_cost(SubParallelComputationGraph const &g,
                    ICostEstimator const &estimator,
                    std::unordered_map<Node, MachineView> const &device_mapping) {
}

// std::size_t num_nodes(OpenSubParallelComputationGraph const &g) {
//   return num_nodes(*g.g);
// }

// std::size_t num_nodes(ParallelComputationGraph const &g) {
//   return num_nodes(g.graph());
// }

// bool is_base_case(SubParallelComputationGraph const &g) {
//   if (holds_alternative<OpenSubParallelComputationGraph>(g)) {
//     return num_nodes(get<OpenSubParallelComputationGraph>(g)) == 1;
//   } else {
//     return num_nodes(get<ParallelComputationGraph>(g)) == 2;
//   }
// }


// We may replace this by having unflattened AST
template<typename T>
std::pair<
  SerialParallelDecomposition,
  SerialParallelDecomposition
> decompose(T const &t) {
  if (t.children.size() == 2) {
    return { widen<SerialParallelDecomposition>(t.children[0]), widen<SerialParallelDecomposition>(t.children[1]) };
  }
  T decompn1 = t;
  decompn1.children.pop_back();
  return { decompn1, widen<SerialParallelDecomposition>(t.children.back()) };
}

GraphSplit get_graph_split(
  SerialParallelDecomposition const &pre_decomposition,
  SerialParallelDecomposition const &post_decomposition) {
  return { get_nodes(pre_decomposition), get_nodes(post_decomposition) };
}

// std::pair<
//   OpenSubParallelComputationGraph, 
//   OpenSubParallelComputationGraph
// > apply_split(OpenSubParallelComputationGraph const &g, GraphSplit const &split) {
  
// }

// std::pair<
//   OpenSubParallelComputationGraph,
//   ParallelComputationGraph
// > apply_split(ParallelComputationGraph const &g, GraphSplit const &split) {

// }

// enum class EdgeDirection {
//   FIRST_TO_SECOND,
//   SECOND_TO_FIRST
// };

std::pair<
  SubParallelComputationGraph,
  SubParallelComputationGraph
> apply_split(SubParallelComputationGraph const &g, GraphSplit const &split) {
  // Need the access to the graph view
  auto g1 = unsafe_view_as_subgraph(g.graph(), split.first);
  auto g2 = unsafe_view_as_subgraph(g.graph(), split.second);

  if (get_cut(g.graph(), split).size() > 0) {
    // Sequential split
    if (get_open_sinks(*g1).size() <= get_open_sources(*g2).size()) {
      // get_open_sinks(*g1).size() should be 1 in perfect sp graphs
      return { get_subgraph(g, split.first, InputSettings::INCLUDE_INPUTS, OutputSettings::EXCLUDE_OUTPUTS),
              get_subgraph(g, split.second, InputSettings::INCLUDE_INPUTS, OutputSettings::INCLUDE_OUTPUTS) };
    } else {
      return { get_subgraph(g, split.first, InputSettings::INCLUDE_INPUTS, OutputSettings::INCLUDE_OUTPUTS),
              get_subgraph(g, split.second, InputSettings::EXCLUDE_INPUTS, OutputSettings::INCLUDE_OUTPUTS) };
    }
  } else {
      // Parallel split
      return { get_subgraph(g, split.first, InputSettings::INCLUDE_INPUTS, OutputSettings::INCLUDE_OUTPUTS),
              get_subgraph(g, split.second, InputSettings::INCLUDE_INPUTS, OutputSettings::INCLUDE_OUTPUTS) };
  }
}

std::vector<std::pair<MachineResource, MachineResource>> get_resource_split(MachineResource const &resource) {
  std::vector<std::pair<MachineResource, MachineResource>> result;
  for (int i = 1; i < resource.num_nodes; ++i) {
    result.push_back({MachineResource(i, resource.num_cpus_per_node, resource.num_gpus_per_node),
                      MachineResource(resource.num_nodes - i, resource.num_cpus_per_node, resource.num_gpus_per_node)});
  }
  return result;
}

// float base_case(OpenSubParallelComputationGraph const &g);
// float base_case(ParallelComputationGraph const &g);

// TODO: Generalize the return type to record strategies
// float internal_optimal_cost(SubParallelComputationGraph const &g, 
//                             ICostEstimator const &cost_estimator, 
//                             SerialParallelDecomposition const &sp_decomposition,
//                             MachineResource const &resource,
//                             std::function<std::unordered_set<MachineView>(PCGOperatorAttrs const &, MachineResource const &)> const &f
//                             ) {
//   if (is_base_case(g)) {
//     // base case
//   } else {
//     // non-base-case 
//   }
// }

struct Strategy {
  Strategy(float runtime, std::unordered_map<Node, MachineView> machine_views)
    : runtime(runtime), machine_views(machine_views) {}

  bool operator<(Strategy const &s) const {
    return runtime < s.runtime;
  }

  static Strategy sequential_combine(Strategy const &s1, Strategy const &s2) {
    return Strategy(s1.runtime + s2.runtime, merge_maps(s1.machine_views, s2.machine_views));
  }

  static Strategy parallel_combine(Strategy const &s1, Strategy const &s2) {
    return Strategy(std::max(s1.runtime, s2.runtime), merge_maps(s1.machine_views, s2.machine_views));
  }

  static Strategy infinity() {
    return Strategy(std::numeric_limits<float>::infinity(), std::unordered_map<Node, MachineView>{});
  }

  float runtime;
  std::unordered_map<Node, MachineView> machine_views;
};

struct OptimalCost {
  OptimalCost(SubParallelComputationGraph const &g, 
              ICostEstimator const &cost_estimator, 
              MachineResource const &resource,
              optional<MachineView> const &source_machine_view, // assume perfect SP
              optional<MachineView> const &sink_machine_view,
              std::function<std::unordered_set<MachineView>(PCGOperatorAttrs const &, MachineResource const &)> const &allowed_machine_views)
    : g(g),
      cost_estimator(cost_estimator),
      resource(resource),
      source_machine_view(source_machine_view),
      sink_machine_view(sink_machine_view),
      allowed_machine_views(allowed_machine_views)
    {}

  template <typename T>
  Strategy operator()(T const &t) const {
    return this->optimal_cost(t);
  }

  Strategy optimal_cost(Serial const &serial) const {
    // return sum(vector_transform([&](variant<Parallel, Node> const &t) { return visit(*this, t); }, serial.children));
    SerialParallelDecomposition pre_decompn, post_decompn;
    std::tie(pre_decompn, post_decompn) = decompose(serial);

    auto subgraphs = apply_split(g, get_graph_split(pre_decompn, post_decompn));
    SubParallelComputationGraph pre_graph = subgraphs.first, post_graph = subgraphs.second;
    
    std::unordered_set<Node> pre_graph_sinks = get_closed_sinks(pre_graph.graph());
    std::unordered_set<Node> post_graph_sources = get_closed_sources(post_graph.graph());

    assert(pre_graph_sinks.size() + post_graph_sources.size() == 1); // assume perfect SP

    Node const &split_point = get_only(set_union(pre_graph_sinks, post_graph_sources));

    Strategy optimal_result = Strategy::infinity();

    for (MachineView const &mv : allowed_machine_views(g.at(split_point), resource)) {
      optional<MachineView> pre_sink_mv = contains(pre_graph_sinks, split_point) ? make_optional(mv) : nullopt;
      optional<MachineView> post_source_mv = contains(post_graph_sources, split_point) ? make_optional(mv) : nullopt;
      minimize(optimal_result,
        Strategy::sequential_combine(
          visit(OptimalCost(pre_graph, cost_estimator, resource, source_machine_view, pre_sink_mv, allowed_machine_views), pre_decompn),
          visit(OptimalCost(post_graph, cost_estimator, resource, post_source_mv, sink_machine_view, allowed_machine_views), post_decompn)));
    }
  }

  Strategy optimal_cost(Parallel const &parallel) const {
    SerialParallelDecomposition decompn1, decompn2;

    std::tie(decompn1, decompn2) = decompose(parallel);

    auto subgraphs = apply_split(g, get_graph_split(decompn1, decompn2));
    SubParallelComputationGraph g1 = subgraphs.first, g2 = subgraphs.second;

    Strategy optimal_result = Strategy::sequential_combine(
      visit(OptimalCost(g1, cost_estimator, resource, source_machine_view, sink_machine_view, allowed_machine_views), decompn1),
      visit(OptimalCost(g2, cost_estimator, resource, source_machine_view, sink_machine_view, allowed_machine_views), decompn2));

    for (auto const &resource_split : get_resource_split(resource)) {
      minimize(optimal_result,
        Strategy::parallel_combine(
          visit(OptimalCost(g1, cost_estimator, resource_split.first, source_machine_view, sink_machine_view, allowed_machine_views), decompn1),
          visit(OptimalCost(g2, cost_estimator, resource_split.second, source_machine_view, sink_machine_view, allowed_machine_views), decompn2)));
    }
  }

  Strategy optimal_cost(Node const &node) const {
    if (source_machine_view) {
      assert(get_closed_sources(g.graph()).empty());
      assert(contains(allowed_machine_views(g.at(node), resource), source_machine_view.value()));
      std::unordered_map<Node, MachineView> mv_map{{node, source_machine_view.value()}};
      return Strategy(estimate_cost(g, cost_estimator, mv_map), mv_map);
    } else if (sink_machine_view) {
      assert(get_closed_sinks(g.graph()).empty());
      assert(contains(allowed_machine_views(g.at(node), resource), sink_machine_view.value()));
      std::unordered_map<Node, MachineView> mv_map{{node, sink_machine_view.value()}};
      return Strategy(estimate_cost(g, cost_estimator, mv_map), mv_map);
    } else {
      Strategy optimal_result = Strategy::infinity();
      for (auto mv : allowed_machine_views(g.at(node), resource)) {
        std::unordered_map<Node, MachineView> mv_map{{node, mv}};
        minimize(optimal_result, Strategy(estimate_cost(g, cost_estimator, mv_map), mv_map));
      }
      return optimal_result;
    }
  }

  SubParallelComputationGraph const &g;
  ICostEstimator const &cost_estimator;
  MachineResource const &resource;
  optional<MachineView> const &source_machine_view;
  optional<MachineView> const &sink_machine_view;
  std::function<std::unordered_set<MachineView>(PCGOperatorAttrs const &, MachineResource const &)> const &allowed_machine_views;
};

// float optimal_cost(ParallelComputationGraph const &g, 
//                    SerialParallelDecomposition const &sp_decomposition, 
//                    std::unordered_set<MachineView> const &allowed_machine_views,
//                    ICostEstimator const &cost_estimator) {
//   return visit(OptimalCost(g, allowed_machine_views, cost_estimator), sp_decomposition);
// }

}