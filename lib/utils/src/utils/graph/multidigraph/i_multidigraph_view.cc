#include "utils/graph/multidigraph/i_multidigraph_view.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

std::unordered_set<DirectedEdge>
    IMultiDiGraphView::query_edges(DirectedEdgeQuery const &q) const {
  return transform(this->query_edges(MultiDiEdgeQuery{q.srcs, q.dsts}),
                   [&](MultiDiEdge const &e) {
                     return DirectedEdge{this->get_multidiedge_src(e),
                                         this->get_multidiedge_dst(e)};
                   });
}

} // namespace FlexFlow
