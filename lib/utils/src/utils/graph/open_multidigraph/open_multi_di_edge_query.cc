#include "utils/graph/open_multidigraph/open_multi_di_edge_query.h"
#include "utils/graph/open_multidigraph/input_multi_di_edge_query.h"
#include "utils/graph/open_multidigraph/output_multi_di_edge_query.h"
#include "utils/graph/multidigraph/multi_di_edge_query.h"

namespace FlexFlow {

OpenMultiDiEdgeQuery open_multidiedge_query_all() {
  return OpenMultiDiEdgeQuery{
    input_multidiedge_query_all(),
    multidiedge_query_all(),
    output_multidiedge_query_all(),
  };
}

} // namespace FlexFlow
