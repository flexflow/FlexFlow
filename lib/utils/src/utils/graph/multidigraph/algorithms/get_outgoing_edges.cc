#include "utils/graph/multidigraph/algorithms/get_outgoing_edges.h"

namespace FlexFlow {

std::unordered_set<MultiDiEdge> get_outgoing_edges(MultiDiGraphView const &g, Node const &n) {
  return g.query_edges(MultiDiEdgeQuery{{n}, query_set<Node>::matchall()});
}

} // namespace FlexFlow
