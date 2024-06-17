#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_DATAFLOW_GRAPH_ALGORITHMS_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_DATAFLOW_GRAPH_ALGORITHMS_H

#include "pcg/dataflow_graph/dataflow_graph.h"

namespace FlexFlow {

template <typename NodeLabel, typename OutputLabel>
std::vector<MultiDiOutput>
    get_inputs(DataflowGraph<NodeLabel, OutputLabel> const &g, Node const &n) {
  std::vector<std::pair<int, MultiDiOutput>> input_edges =
      transform(as_vector(get_incoming_edges(g.get_raw_graph(),
                                             std::unordered_set<Node>{n})),
                [&](MultiDiEdge const &e) {
                  int idx = g.idx_for_port(e.dst_idx);
                  MultiDiOutput val = static_cast<MultiDiOutput>(e);
                  return std::make_pair(idx, val);
                });

  return vector_from_indexed_set(input_edges);
}

template <typename NodeLabel, typename OutputLabel>
std::vector<MultiDiOutput>
    get_outputs(DataflowGraph<NodeLabel, OutputLabel> const &g, Node const &n) {
  return g.get_output_map().at(n);
}

template <typename NodeLabel, typename OutputLabel>
std::vector<Node> topological_ordering(DataflowGraph<NodeLabel, OutputLabel> const &g) {
  return get_topological_ordering(g.get_raw_graph());
}

} // namespace FlexFlow

#endif
