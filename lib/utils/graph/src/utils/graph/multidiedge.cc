#include "utils/graph/multidiedge.h"

namespace FlexFlow {

OutputMultiDiEdgeQuery OutputMultiDiEdgeQuery::all() {
  return {matchall<Node>(), matchall<NodePort>()};
}

OutputMultiDiEdgeQuery OutputMultiDiEdgeQuery::none() {
  return {{}, {}};
}

InputMultiDiEdgeQuery InputMultiDiEdgeQuery::all() {
  return {matchall<Node>(), matchall<NodePort>()};
}

InputMultiDiEdgeQuery InputMultiDiEdgeQuery::none() {
  return {{}, {}};
}

MultiDiEdgeQuery
    MultiDiEdgeQuery::with_src_nodes(query_set<Node> const &nodes) const {
  MultiDiEdgeQuery e = *this;
  if (!is_matchall(e.srcs)) {
    throw mk_runtime_error("Expected matchall previous value");
  }
  e.srcs = nodes;
  return e;
}

MultiDiEdgeQuery
    MultiDiEdgeQuery::with_dst_nodes(query_set<Node> const &nodes) const {
  MultiDiEdgeQuery e = *this;
  if (!is_matchall(e.dsts)) {
    throw mk_runtime_error("Expected matchall previous value");
  }
  e.dsts = nodes;
  return e;
}

MultiDiEdgeQuery
    MultiDiEdgeQuery::with_src_idxs(query_set<NodePort> const &idxs) const {
  MultiDiEdgeQuery e{*this};
  if (!is_matchall(e.srcIdxs)) {
    throw mk_runtime_error("Expected matchall previous value");
  }
  e.srcIdxs = idxs;
  return e;
}

MultiDiEdgeQuery
    MultiDiEdgeQuery::with_dst_idxs(query_set<NodePort> const &idxs) const {
  MultiDiEdgeQuery e = *this;
  if (!is_matchall(e.dstIdxs)) {
    throw mk_runtime_error("Expected matchall previous value");
  }
  e.dstIdxs = idxs;
  return e;
}

MultiDiEdgeQuery MultiDiEdgeQuery::all() {
  return {matchall<Node>(),
          matchall<Node>(),
          matchall<NodePort>(),
          matchall<NodePort>()};
}

MultiDiEdgeQuery MultiDiEdgeQuery::none() {
  return {{}, {}, {}, {}};
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

  std::unordered_set<NodePort> srcIdxs_t1;
  if (is_matchall(lhs.srcIdxs) && !is_matchall(rhs.srcIdxs)) {
    srcIdxs_t1 = allowed_values(rhs.srcIdxs);
  } else if (!is_matchall(lhs.srcIdxs) && is_matchall(rhs.srcIdxs)) {
    srcIdxs_t1 = allowed_values(lhs.srcIdxs);
  } else if (!is_matchall(lhs.srcIdxs) && !is_matchall(rhs.srcIdxs)) {
    srcIdxs_t1 = allowed_values(query_intersection(lhs.srcIdxs, rhs.srcIdxs));
  }

  std::unordered_set<NodePort> dstIdxs_t1;
  if (is_matchall(lhs.dstIdxs) && !is_matchall(rhs.dstIdxs)) {
    dstIdxs_t1 = allowed_values(rhs.dstIdxs);
  } else if (!is_matchall(lhs.dstIdxs) && is_matchall(rhs.dstIdxs)) {
    dstIdxs_t1 = allowed_values(lhs.dstIdxs);
  } else if (!is_matchall(lhs.dstIdxs) && !is_matchall(rhs.dstIdxs)) {
    dstIdxs_t1 = allowed_values(query_intersection(lhs.dstIdxs, rhs.dstIdxs));
  }

  MultiDiEdgeQuery e = MultiDiEdgeQuery::all();
  e.srcs = srcs_t1;
  e.dsts = dsts_t1;
  e.srcIdxs = srcIdxs_t1;
  e.dstIdxs = dstIdxs_t1;
  return e;
}

} // namespace FlexFlow
