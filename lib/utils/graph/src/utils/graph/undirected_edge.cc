#include "utils/graph/undirected_edge.h"

namespace FlexFlow {

UndirectedEdge::UndirectedEdge(Node const &n1, Node const &n2)
    : smaller(std::min(n1, n2)), bigger(std::max(n1, n2)) {}

bool is_connected_to(UndirectedEdge const &e, Node const &n) {
  return e.bigger == n || e.smaller == n;
}

UndirectedEdgeQuery UndirectedEdgeQuery::all() {
  return {matchall<Node>()};
}

UndirectedEdgeQuery query_intersection(UndirectedEdgeQuery const &lhs,
                                       UndirectedEdgeQuery const &rhs) {
  return {
      query_intersection(lhs.nodes, rhs.nodes),
  };
}

}