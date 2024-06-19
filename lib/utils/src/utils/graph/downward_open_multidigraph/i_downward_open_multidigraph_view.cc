#include "utils/graph/downward_open_multidigraph/i_downward_open_multidigraph_view.h"

namespace FlexFlow {

std::unordered_set<OpenMultiDiEdge>
    IDownwardOpenMultiDiGraphView::query_edges(OpenMultiDiEdgeQuery const &q) const {
  NOT_IMPLEMENTED();
  //return widen<OpenMultiDiEdge>(
  //    this->query_edges(DownwardOpenMultiDiEdgeQuery{q.output_edge_query,
  //                                                   q.standard_edge_query}));
}

} // namespace FlexFlow
