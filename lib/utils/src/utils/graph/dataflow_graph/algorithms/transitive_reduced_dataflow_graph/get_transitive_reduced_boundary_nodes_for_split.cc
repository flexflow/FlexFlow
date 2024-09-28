#include "utils/graph/dataflow_graph/algorithms/transitive_reduced_dataflow_graph/get_transitive_reduced_boundary_nodes_for_split.h"
#include "utils/graph/dataflow_graph/algorithms/transitive_reduced_dataflow_graph/get_transitive_reduced_edges_across_split.h"

namespace FlexFlow {

SplitBoundaryNodes get_transitive_reduced_boundary_nodes_for_split(TransitiveReducedDataflowGraphView const &tr_g,
                                                                   BinarySeriesSplit const &split) {
  std::unordered_set<DataflowEdge> edges = get_transitive_reduced_edges_across_split(tr_g, split);

  std::unordered_set<Node> src_boundary_nodes 
    = transform(edges,
                [](DataflowEdge const &e) { return e.src.node; });

  std::unordered_set<Node> dst_boundary_nodes 
    = transform(edges,
                [](DataflowEdge const &e) { return e.dst.node; });

  return SplitBoundaryNodes{
    /*pre_split_boundary=*/src_boundary_nodes,
    /*post_split_boundary=*/dst_boundary_nodes,
  };
}

} // namespace FlexFlow
