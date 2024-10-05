#include "utils/graph/dataflow_graph/algorithms/transitive_reduced_dataflow_graph/get_transitive_reduced_edges_across_split.h"
#include "utils/containers/flatmap.h"
#include "utils/graph/dataflow_graph/algorithms/get_dataflow_edges_from_node_to_node.h"
#include "utils/graph/digraph/algorithms/get_edges_from_subgraph_to_subgraph.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_series_split.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.h"

namespace FlexFlow {

std::unordered_set<DataflowEdge> get_transitive_reduced_edges_across_split(
    TransitiveReducedDataflowGraphView const &tr_g,
    BinarySeriesSplit const &split) {
  std::unordered_set<Node> src_subgraph =
      unordered_set_of(get_leaves(get_left_child(split)));
  std::unordered_set<Node> dst_subgraph =
      unordered_set_of(get_leaves(get_right_child(split)));

  std::unordered_set<DirectedEdge> raw_edges =
      get_edges_from_subgraph_to_subgraph(
          tr_g.transitive_reduction, src_subgraph, dst_subgraph);

  return flatmap(raw_edges, [&](DirectedEdge const &e) {
    return get_dataflow_edges_from_node_to_node(
        tr_g.full_dataflow_graph, e.src, e.dst);
  });
}

} // namespace FlexFlow
