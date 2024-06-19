#include "utils/graph/multidigraph/multi_di_edge_query.h"

namespace FlexFlow {

MultiDiEdgeQuery multidiedge_query_all() {
  return MultiDiEdgeQuery{matchall<Node>(),
          matchall<Node>()};
}

MultiDiEdgeQuery multidiedge_query_none() {
  return MultiDiEdgeQuery{query_set<Node>({}),
          query_set<Node>({})};
}

MultiDiEdgeQuery query_intersection(MultiDiEdgeQuery const &lhs,
                                    MultiDiEdgeQuery const &rhs) {
  std::unordered_set<Node> srcs_t1;
  if (is_matchall(lhs.srcs) && !is_matchall(rhs.srcs)) {
    srcs_t1 = allowed_values(rhs.srcs);
  } else if (!is_matchall(lhs.srcs) && is_matchall(rhs.srcs)) {
    srcs_t1 = allowed_values(lhs.srcs);
  } else if (!is_matchall(lhs.srcs) && !is_matchall(rhs.srcs)) {
    srcs_t1 = allowed_values(query_intersection(lhs.srcs, rhs.srcs));
  }

  std::unordered_set<Node> dsts_t1;
  if (is_matchall(lhs.dsts) && !is_matchall(rhs.dsts)) {
    dsts_t1 = allowed_values(rhs.dsts);
  } else if (!is_matchall(lhs.dsts) && is_matchall(rhs.dsts)) {
    dsts_t1 = allowed_values(lhs.dsts);
  } else if (!is_matchall(lhs.dsts) && !is_matchall(rhs.dsts)) {
    dsts_t1 = allowed_values(query_intersection(lhs.dsts, rhs.dsts));
  }

  MultiDiEdgeQuery e = multidiedge_query_all();
  e.srcs = srcs_t1;
  e.dsts = dsts_t1;
  return e;
}

MultiDiEdgeQuery query_union(MultiDiEdgeQuery const &,
                             MultiDiEdgeQuery const &) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
