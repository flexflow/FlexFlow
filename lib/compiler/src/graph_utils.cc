#include "graph_utils.h"

namespace FlexFlow {

SerialParallelDecomposition
    get_serial_parallel_decomposition(ParallelComputationGraph const &pcg) {
  return get_serial_parallel_decomposition(as_digraph(pcg));
}

std::vector<MultiDiEdge>
    get_sorted_node_input_edges(ParallelComputationGraph const &pcg,
                                Node const &n) {
  std::unordered_map<NodePort, std::unordered_set<MultiDiEdge>> incoming_edges =
      get_incoming_edges_by_idx(pcg, n);

  std::vector<MultiDiEdge> result;
  for (auto const &p_id_edge_set : incoming_edges) {
    result.push_back(get_only(p_id_edge_set.second));
  }

  return result;
}

std::unordered_map<MultiDiEdge, ParallelTensorShape>
    infer_tensor_shapes(ParallelComputationGraph const &pcg) {
  std::unordered_map<MultiDiEdge, ParallelTensorShape> result;
  for (Node const &n : get_topological_ordering(pcg)) {
    PCGOperatorAttrs op = pcg.value().at(n);

    std::vector<ParallelTensorShape> input_tensor_shapes =
        vector_transform([&](MultiDiEdge const &e) { return result.at(e); },
                         get_sorted_node_input_edges(pcg, n));

    std::vector<ParallelTensorShape> output_tensor_shapes =
        get_output_shapes(op, input_tensor_shapes);

    auto outgoing_edges = get_outgoing_edges_by_idx(pcg, n);

    int i = 0;

    for (auto const &[node_port, edges] : outgoing_edges) {
      for (MultiDiEdge const &e : edges) {
        result.insert({e, output_tensor_shapes[i++]});
      }
    }
  }

  assert(result.size() == get_edges(pcg).size());

  return result;
}

/* template <typename NodeLabel, */
/*           typename EdgeLabel, */
/*           typename InputLabel, */
/*           typename OutputLabel> */
/* LabelledOpenMultiDiGraph<NodeLabel, EdgeLabel, InputLabel, OutputLabel> */
/*     get_subgraph(LabelledOpenMultiDiGraph<NodeLabel, */
/*                                           EdgeLabel, */
/*                                           InputLabel, */
/*                                           OutputLabel> const &g, */
/*                  std::unordered_set<Node> const &nodes, */
/*                  InputSettings input_settings, */
/*                  OutputSettings output_settings) { */

/*   auto iview = LabelledOpenMultiDiGraphView<NodeLabel, */
/*                                             EdgeLabel, */
/*                                             InputLabel, */
/*                                             OutputLabel>(g) */
/*                    .unsafe(); */

/*   if (input_settings == InputSettings::INCLUDE_INPUTS && */
/*       output_settings == OutputSettings::INCLUDE_OUTPUTS) { */
/*     LabelledOpenMultiDiSubgraphView<NodeLabel, */
/*                                     EdgeLabel, */
/*                                     InputLabel, */
/*                                     OutputLabel> */
/*         subgraph_view(*iview, nodes); */
/*     return materialize_labelled_openmultidigraph_view(subgraph_view); */
/*   } else if (input_settings == InputSettings::INCLUDE_INPUTS && */
/*              output_settings == OutputSettings::EXCLUDE_OUTPUTS) { */
/*     LabelledUpwardMultiDiSubgraphView<NodeLabel, EdgeLabel, InputLabel> */
/*         subgraph_view(*iview, nodes); */
/*     return materialize_labelled_openmultidigraph_view( */
/*         view_as_labelled_open_multidisubgraph(subgraph_view)); */
/*   } else if (input_settings == InputSettings::EXCLUDE_INPUTS && */
/*              output_settings == OutputSettings::INCLUDE_OUTPUTS) { */
/*     LabelledDownwardMultiDiSubgraphView<NodeLabel, EdgeLabel, OutputLabel> */
/*         subgraph_view(*iview, nodes); */
/*     return materialize_labelled_openmultidigraph_view( */
/*         view_as_labelled_open_multidisubgraph(subgraph_view)); */
/*   } else { */
/*     LabelledMultiDiSubgraphView<NodeLabel, EdgeLabel> subgraph_view(*iview,
 */
/*                                                                     nodes);
 */
/*     return materialize_labelled_openmultidigraph_view( */
/*         view_as_labelled_open_multidisubgraph<NodeLabel, */
/*                                               EdgeLabel, */
/*                                               InputLabel, */
/*                                               OutputLabel>(subgraph_view));
 */
/*   } */
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
  return set_union(
      transform(serial.children, [](variant<Parallel, Node> const child) {
        return visit(GetNodes{}, child);
      }));
}

std::unordered_set<Node> get_nodes(Parallel const &parallel) {
  return set_union(
      transform(parallel.children, [](variant<Serial, Node> const child) {
        return visit(GetNodes{}, child);
      }));
}

std::unordered_set<Node> get_nodes(Node const &node) {
  return {node};
}

} // namespace FlexFlow
