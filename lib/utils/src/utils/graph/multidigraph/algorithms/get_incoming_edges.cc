#include "utils/graph/multidigraph/algorithms/get_incoming_edges.h"

namespace FlexFlow {

std::unordered_set<MultiDiEdge> get_incoming_edges(MultiDiGraphView const &g,
                                                   Node const &n) {
  return g.query_edges(MultiDiEdgeQuery{query_set<Node>::matchall(), {n}});
}

} // namespace FlexFlow
