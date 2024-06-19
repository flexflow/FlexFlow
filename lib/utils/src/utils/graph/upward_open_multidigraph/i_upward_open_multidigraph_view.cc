#include "utils/graph/upward_open_multidigraph/i_upward_open_multidigraph_view.h"
#include "utils/graph/upward_open_multidigraph/upward_open_multi_di_edge.h"
#include "utils/containers.h"

namespace FlexFlow {

std::unordered_set<OpenMultiDiEdge>
    IUpwardOpenMultiDiGraphView::query_edges(OpenMultiDiEdgeQuery const &q) const {

  std::unordered_set<UpwardOpenMultiDiEdge> queried = this->query_edges(
      UpwardOpenMultiDiEdgeQuery{q.input_edge_query, q.standard_edge_query});

  return transform(queried, [](UpwardOpenMultiDiEdge const &upward_e) { return open_multidiedge_from_upward_open(upward_e); });
}

} // namespace FlexFlow
