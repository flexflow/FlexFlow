#include "utils/graph/undirected/undirected_edge_query.h"

namespace FlexFlow {

UndirectedEdgeQuery undirected_edge_query_all() {
  return UndirectedEdgeQuery{matchall<Node>()};
}

UndirectedEdgeQuery query_intersection(UndirectedEdgeQuery const &lhs,
                                       UndirectedEdgeQuery const &rhs) {
  return UndirectedEdgeQuery{
      query_intersection(lhs.nodes, rhs.nodes),
  };
}

} // namespace FlexFlow
