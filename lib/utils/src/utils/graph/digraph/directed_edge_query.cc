#include "utils/graph/digraph/directed_edge_query.h"

namespace FlexFlow {

DirectedEdgeQuery directed_edge_query_all() {
  return DirectedEdgeQuery{matchall<Node>(), matchall<Node>()};
}

bool matches_edge(DirectedEdgeQuery const &q, DirectedEdge const &e) {
  return includes(q.srcs, e.src) && includes(q.dsts, e.dst);
}

DirectedEdgeQuery query_intersection(DirectedEdgeQuery const &lhs,
                                     DirectedEdgeQuery const &rhs) {
  std::unordered_set<Node> result_srcs;
  if (is_matchall(lhs.srcs) && !is_matchall(rhs.srcs)) {
    result_srcs = allowed_values(rhs.srcs);
  } else if (!is_matchall(lhs.srcs) && is_matchall(rhs.srcs)) {
    result_srcs = allowed_values(lhs.srcs);
  } else if (!is_matchall(lhs.srcs) && !is_matchall(rhs.srcs)) {
    result_srcs = allowed_values(query_intersection(lhs.srcs, rhs.srcs));
  }

  std::unordered_set<Node> result_dsts;
  if (is_matchall(lhs.dsts) && !is_matchall(rhs.dsts)) {
    result_dsts = allowed_values(rhs.dsts);
  } else if (!is_matchall(lhs.dsts) && is_matchall(rhs.dsts)) {
    result_dsts = allowed_values(lhs.dsts);
  } else if (!is_matchall(lhs.dsts) && !is_matchall(rhs.dsts)) {
    result_dsts = allowed_values(query_intersection(lhs.dsts, rhs.dsts));
  }

  return DirectedEdgeQuery{result_srcs, result_dsts};
}

} // namespace FlexFlow
