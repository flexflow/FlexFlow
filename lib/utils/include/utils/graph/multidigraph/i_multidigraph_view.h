#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_MULTIDIGRAPH_I_MULTIDIGRAPH_VIEW_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_MULTIDIGRAPH_I_MULTIDIGRAPH_VIEW_H

#include "utils/graph/digraph/i_digraph_view.h"
#include "utils/graph/multidigraph/multi_di_edge.dtg.h"
#include "utils/graph/multidigraph/multi_di_edge_query.dtg.h"

namespace FlexFlow {

struct IMultiDiGraphView : virtual public IDiGraphView {
  using Edge = MultiDiEdge;
  using EdgeQuery = MultiDiEdgeQuery;

  virtual std::unordered_set<Edge> query_edges(EdgeQuery const &) const = 0;
  std::unordered_set<DirectedEdge>
      query_edges(DirectedEdgeQuery const &) const override final;
  virtual ~IMultiDiGraphView() = default;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IMultiDiGraphView);


} // namespace FlexFlow

#endif
