#include "utils/graph/undirected/undirected_edge_query.h"

namespace FlexFlow {

UndirectedEdgeQuery undirected_edge_query_all() {
  return UndirectedEdgeQuery{matchall<Node>()};
}

bool matches_edge(UndirectedEdgeQuery const &q, UndirectedEdge const &e) {
  return includes(q.nodes, e.bigger) && includes(q.nodes, e.smaller);
}

UndirectedEdgeQuery query_intersection(UndirectedEdgeQuery const &lhs,
                                       UndirectedEdgeQuery const &rhs) {
  return UndirectedEdgeQuery{
      query_intersection(lhs.nodes, rhs.nodes),
  };
}

} // namespace FlexFlow
