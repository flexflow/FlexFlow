#include "utils/graph/multidigraph/i_multidigraph_view.h"

namespace FlexFlow {

std::unordered_set<DirectedEdge>
    IMultiDiGraphView::query_edges(DirectedEdgeQuery const &q) const {
  return transform(
      query_edges(MultiDiEdgeQuery{
          q.srcs, q.dsts}),
      [](MultiDiEdge const &e) {
        return DirectedEdge{e.src, e.dst};
      });
}

} // namespace FlexFlow
