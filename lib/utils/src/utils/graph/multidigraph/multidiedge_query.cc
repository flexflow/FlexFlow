#include "utils/graph/multidigraph/multidiedge_query.h"

namespace FlexFlow {

MultiDiEdgeQuery multidiedge_query_all() {
  return MultiDiEdgeQuery{matchall<Node>(), matchall<Node>()};
}

MultiDiEdgeQuery multidiedge_query_none() {
  return MultiDiEdgeQuery{
      query_set<Node>::match_none(),
      query_set<Node>::match_none(),
  };
}

} // namespace FlexFlow
