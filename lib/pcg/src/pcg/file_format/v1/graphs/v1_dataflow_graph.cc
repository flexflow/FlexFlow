#include "pcg/file_format/v1/graphs/v1_dataflow_graph.h"
#include "utils/containers/enumerate.h"
#include "utils/containers/sorted.h"
#include "utils/containers/values.h"
#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/node/algorithms.h"
#include "utils/integer_conversions.h"
#include "utils/bidict/algorithms/bidict_from_enumerating.h"

namespace FlexFlow {

V1DataflowGraph to_v1(DataflowGraphView const &g) {
  bidict<int, Node> node_enumeration_bidict = bidict_from_enumerating(get_nodes(g));
  std::unordered_map<Node, int> node_enumeration = node_enumeration_bidict.reversed().as_unordered_map();
  return to_v1(g, node_enumeration);
}

V1DataflowGraph to_v1(DataflowGraphView const &g,
                      std::unordered_map<Node, int> const &nodes) {
  std::unordered_set<V1GraphEdge> edges;
  for (DataflowEdge const &e : get_edges(g)) {
    edges.insert(V1GraphEdge{nodes.at(e.src.node),
                             e.src.idx,
                             nodes.at(e.dst.node),
                             e.dst.idx});
  }

  return V1DataflowGraph{
      sorted(values(nodes)),
      edges,
  };
}

} // namespace FlexFlow
