#include "utils/graph/open_multidigraph/output_multi_di_edge_query.h"

namespace FlexFlow {

OutputMultiDiEdgeQuery output_multidiedge_query_all() {
  return OutputMultiDiEdgeQuery{query_set<Node>::matchall()};
}

OutputMultiDiEdgeQuery output_multidiedge_query_none() {
  return OutputMultiDiEdgeQuery{query_set<Node>::match_none()};
}

} // namespace FlexFlow
