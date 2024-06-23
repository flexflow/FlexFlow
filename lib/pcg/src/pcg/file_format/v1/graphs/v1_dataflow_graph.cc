#include "pcg/file_format/v1/graphs/v1_dataflow_graph.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/dataflow_graph/algorithms.h"

namespace FlexFlow {

V1DataflowGraph to_v1(DataflowGraphView const &g) {
  return to_v1(g, enumerate(get_nodes(g)).reversed());
}

V1DataflowGraph to_v1(DataflowGraphView const &g,
                      bidict<Node, size_t> const &nodes) {
  std::unordered_set<V1GraphEdge> edges;
  for (DataflowEdge const &e : get_edges(g)) {
    edges.insert(V1GraphEdge{nodes.at_l(e.src.node),
                             e.src.idx,
                             nodes.at_l(e.dst.node),
                             e.dst.idx});
  }

  return V1DataflowGraph{
      sorted(values(nodes)),
      edges,
  };
}


} // namespace FlexFlow
