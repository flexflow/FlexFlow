#include "graph_utils.h"

namespace FlexFlow {

SerialParallelDecomposition
    get_serial_parallel_decomposition(OptimizerPCG const &pcg) {
  return get_serial_parallel_decomposition(
      unsafe_view_as_digraph(MultiDiGraphView(pcg)));
}

std::vector<MultiDiEdge> get_sorted_node_input_edges(OptimizerPCG const &pcg,
                                                     Node const &n) {
  std::unordered_map<std::size_t, std::unordered_set<MultiDiEdge>>
      incoming_edges = get_incoming_edges_by_idx(MultiDiGraphView(pcg), n);

  std::vector<MultiDiEdge> result;
  for (std::size_t i = 0; i < incoming_edges.size(); i++) {
    result.push_back(get_only(incoming_edges.at(i)));
  }

  return result;
}

std::unordered_map<MultiDiEdge, ParallelTensorShape>
    infer_tensor_shapes(OptimizerPCG const &pcg) {
  std::unordered_map<MultiDiEdge, ParallelTensorShape> result;
  for (Node const &n : get_topological_ordering(MultiDiGraphView(pcg))) {
    PCGOperatorAttrs op = pcg.at(n);

    std::vector<ParallelTensorShape> input_tensor_shapes =
        vector_transform([&](MultiDiEdge const &e) { return result.at(e); },
                         get_sorted_node_input_edges(pcg, n));

    std::vector<ParallelTensorShape> output_tensor_shapes =
        get_output_shapes(op, input_tensor_shapes);

    auto outgoing_edges = get_outgoing_edges_by_idx(MultiDiGraphView(pcg), n);

    for (std::size_t i = 0; i < output_tensor_shapes.size(); i++) {
      if (contains_key(outgoing_edges, i)) {
        for (MultiDiEdge const &e : outgoing_edges.at(i)) {
          result.insert({e, output_tensor_shapes[i]});
        }
      }
    }
  }

  assert(result.size() == get_edges(MultiDiGraphView(pcg)).size());

  return result;
}

template <typename NodeLabel,
          typename EdgeLabel,
          typename InputLabel = EdgeLabel,
          typename OutputLabel = EdgeLabel>
LabelledOpenMultiDiGraph<NodeLabel, EdgeLabel, InputLabel, OutputLabel>
    get_subgraph(LabelledOpenMultiDiGraph<NodeLabel,
                                          EdgeLabel,
                                          InputLabel,
                                          OutputLabel> const &g,
                 std::unordered_set<Node> const &nodes,
                 InputSettings input_settings,
                 OutputSettings output_settings) {

  auto iview = LabelledOpenMultiDiGraphView<NodeLabel,
                                            EdgeLabel,
                                            InputLabel,
                                            OutputLabel>(g)
                   .unsafe();

  if (input_settings == InputSettings::INCLUDE_INPUTS &&
      output_settings == OutputSettings::INCLUDE_OUTPUTS) {
    LabelledOpenMultiDiSubgraphView<NodeLabel,
                                    EdgeLabel,
                                    InputLabel,
                                    OutputLabel>
        subgraph_view(*iview, nodes);
    return materialize_labelled_openmultidigraph_view(subgraph_view);
  } else if (input_settings == InputSettings::INCLUDE_INPUTS &&
             output_settings == OutputSettings::EXCLUDE_OUTPUTS) {
    LabelledUpwardMultiDiSubgraphView<NodeLabel, EdgeLabel, InputLabel>
        subgraph_view(*iview, nodes);
    return materialize_labelled_openmultidigraph_view(
        view_as_labelled_open_multidisubgraph(subgraph_view));
  } else if (input_settings == InputSettings::EXCLUDE_INPUTS &&
             output_settings == OutputSettings::INCLUDE_OUTPUTS) {
    LabelledDownwardMultiDiSubgraphView<NodeLabel, EdgeLabel, OutputLabel>
        subgraph_view(*iview, nodes);
    return materialize_labelled_openmultidigraph_view(
        view_as_labelled_open_multidisubgraph(subgraph_view));
  } else {
    LabelledMultiDiSubgraphView<NodeLabel, EdgeLabel> subgraph_view(*iview,
                                                                    nodes);
    return materialize_labelled_openmultidigraph_view(
        view_as_labelled_open_multidisubgraph<NodeLabel,
                                              EdgeLabel,
                                              InputLabel,
                                              OutputLabel>(subgraph_view));
  }
}

} // namespace FlexFlow
