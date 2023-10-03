#include "utils/graph/diedge.h"

namespace FlexFlow {

DirectedEdgeQuery DirectedEdgeQuery::all() {
  return {matchall<Node>(), matchall<Node>()};
}

bool matches_edge(DirectedEdgeQuery const &q, DirectedEdge const &e) {
  return includes(q.srcs, e.src) && includes(q.dsts, e.dst);
}

DirectedEdgeQuery query_intersection(DirectedEdgeQuery const &lhs,
                                     DirectedEdgeQuery const &rhs) {
  std::unordered_set<Node> srcs_tl;
  if (is_matchall(lhs.srcs) && !is_matchall(rhs.srcs)) {
    srcs_tl = allowed_values(rhs.srcs);
  } else if (!is_matchall(lhs.srcs) && is_matchall(rhs.srcs)) {
    srcs_tl = allowed_values(lhs.srcs);
  } else if (!is_matchall(lhs.srcs) && !is_matchall(rhs.srcs)) {
    srcs_tl = allowed_values(query_intersection(lhs.srcs, rhs.srcs));
  }

  std::unordered_set<Node> dsts_tl;
  if (is_matchall(lhs.dsts) && !is_matchall(rhs.dsts)) {
    dsts_tl = allowed_values(rhs.dsts);
  } else if (!is_matchall(lhs.dsts) && is_matchall(rhs.dsts)) {
    dsts_tl = allowed_values(lhs.dsts);
  } else if (!is_matchall(lhs.dsts) && !is_matchall(rhs.dsts)) {
    dsts_tl = allowed_values(query_intersection(lhs.dsts, rhs.dsts));
  }

  DirectedEdgeQuery result = DirectedEdgeQuery::all();
  result.srcs = srcs_tl;
  result.dsts = dsts_tl;
  return result;
}

}