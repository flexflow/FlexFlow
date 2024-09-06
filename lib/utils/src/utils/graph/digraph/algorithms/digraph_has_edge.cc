#include "utils/graph/digraph/algorithms/digraph_has_edge.h"

namespace FlexFlow {

bool digraph_has_edge(DiGraphView const &g, DirectedEdge const &e) {
  return !g.query_edges(DirectedEdgeQuery{
                            query_set<Node>{e.src},
                            query_set<Node>{e.dst},
                        })
              .empty();
}

} // namespace FlexFlow
