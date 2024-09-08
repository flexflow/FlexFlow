#include "compiler/series_parallel/get_computation_graph_series_parallel_decomposition.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "pcg/computation_graph.h"
#include "pcg/computation_graph/computation_graph_edge.h"
#include "utils/graph/digraph/algorithms/digraph_as_dot.h"
#include "utils/graph/digraph/algorithms/materialize_digraph_view.h"
#include "utils/graph/digraph/algorithms/transitive_reduction.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/series_parallel/get_series_parallel_decomposition.h"
#include "utils/record_formatter.h"

namespace FlexFlow {

std::string render_preprocessed_computation_graph_for_sp_decomposition(
    ComputationGraph const &cg) {
  std::unordered_set<layer_guid_t> weight_and_input_layers =
      filter(get_layers(cg), [&](layer_guid_t const &l) {
        ComputationGraphOpAttrs op_attrs = get_layer_attrs(cg, l).attrs;
        return op_attrs.has<WeightAttrs>() || op_attrs.has<InputAttrs>();
      });

  std::unordered_set<layer_guid_t> weight_and_input_layer_successors =
      get_subgraph_successors(cg, weight_and_input_layers);

  // dot has is incapable of rendering the number of edges in the all-to-all
  // connection, so for visualization purposes we instead insert a "fake" node
  // to reduce the n^2 edges to 2*n edges
  DiGraph preprocessed_digraph =
      materialize_digraph_view<AdjacencyDiGraph>(cg.raw_graph);
  Node fake_node = preprocessed_digraph.add_node();
  for (layer_guid_t const &src : weight_and_input_layers) {
    preprocessed_digraph.add_edge(DirectedEdge{src.raw_node, fake_node});
  }
  for (layer_guid_t const &dst : weight_and_input_layer_successors) {
    preprocessed_digraph.add_edge(DirectedEdge{fake_node, dst.raw_node});
  }

  std::function<std::string(Node const &)> get_node_label =
      [&](Node const &n) -> std::string {
    if (n == fake_node) {
      return "FAKE";
    }
    LayerAttrs a = cg.raw_graph.at(n);
    RecordFormatter r = as_dot(a.attrs);

    if (a.name.has_value()) {
      RecordFormatter rr;
      rr << "Name" << a.name.value();
      r << rr;
    }

    std::ostringstream oss;
    oss << r;
    return oss.str();
  };
  std::string preprocessed_dot = digraph_as_dot(
      transitive_reduction(preprocessed_digraph), get_node_label);

  return preprocessed_dot;
}

std::optional<SeriesParallelDecomposition>
    get_computation_graph_series_parallel_decomposition(
        ComputationGraph const &cg) {

  {
    DiGraphView unpreprocessed_digraph = cg.raw_graph;
    std::optional<SeriesParallelDecomposition> unpreprocessed_sp_decomposition = get_series_parallel_decomposition(unpreprocessed_digraph);
    if (unpreprocessed_sp_decomposition.has_value()) {
      return unpreprocessed_sp_decomposition.value();
    }
  }

  DiGraphView preprocessed_digraph = [&] {
    std::unordered_set<layer_guid_t> weight_and_input_layers =
        filter(get_layers(cg), [&](layer_guid_t const &l) {
          ComputationGraphOpAttrs op_attrs = get_layer_attrs(cg, l).attrs;
          return op_attrs.has<WeightAttrs>() || op_attrs.has<InputAttrs>();
        });

    std::unordered_set<layer_guid_t> weight_and_input_layer_successors =
        get_subgraph_successors(cg, weight_and_input_layers);

    DiGraph digraph = materialize_digraph_view<AdjacencyDiGraph>(cg.raw_graph);
    for (layer_guid_t const &src : weight_and_input_layers) {
      for (layer_guid_t const &dst : weight_and_input_layer_successors) {
        digraph.add_edge(DirectedEdge{src.raw_node, dst.raw_node});
      }
    }

    return digraph;
  }();

  return get_series_parallel_decomposition(preprocessed_digraph);
}

} // namespace FlexFlow
