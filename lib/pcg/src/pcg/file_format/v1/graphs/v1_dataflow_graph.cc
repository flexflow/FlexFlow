#include "pcg/file_format/v1/graphs/v1_dataflow_graph.h"
#include "utils/containers/values.h"
#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/node/algorithms.h"
#include "utils/integer_conversions.h"
#include "utils/containers/enumerate.h"

namespace FlexFlow {

V1DataflowGraph to_v1(DataflowGraphView const &g) {
  return to_v1(g, enumerate(get_nodes(g)).reversed());
}

V1DataflowGraph to_v1(DataflowGraphView const &g,
                      std::unordered_map<Node, size_t> const &nodes) {
  std::unordered_set<V1GraphEdge> edges;
  for (DataflowEdge const &e : get_edges(g)) {
    edges.insert(V1GraphEdge{nodes.at(e.src.node),
                             size_t_from_int(e.src.idx),
                             nodes.at(e.dst.node),
                             size_t_from_int(e.dst.idx)});
  }

  return V1DataflowGraph{
      sorted(values(nodes)),
      edges,
  };
}

} // namespace FlexFlow
