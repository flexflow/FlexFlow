#include "utils/graph/upward_open_multidigraph/i_upward_open_multidigraph_view.h"

namespace FlexFlow {

std::unordered_set<OpenMultiDiEdge>
    IUpwardOpenMultiDiGraphView::query_edges(OpenMultiDiEdgeQuery const &q) const {
  return widen<OpenMultiDiEdge>(this->query_edges(
      UpwardOpenMultiDiEdgeQuery{q.input_edge_query, q.standard_edge_query}));
}

} // namespace FlexFlow
