#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_MULTIDIGRAPH_I_OPEN_MULTIDIGRAPH_VIEW_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_MULTIDIGRAPH_I_OPEN_MULTIDIGRAPH_VIEW_H

#include "utils/graph/multidigraph/i_multidigraph_view.h"
#include "utils/graph/open_multidigraph/open_multi_di_edge.dtg.h"
#include "utils/graph/open_multidigraph/open_multi_di_edge_query.dtg.h"

namespace FlexFlow {

struct IOpenMultiDiGraphView : virtual public IMultiDiGraphView {
  virtual std::unordered_set<OpenMultiDiEdge>
      query_edges(OpenMultiDiEdgeQuery const &) const = 0;
  virtual std::unordered_set<MultiDiEdge>
      query_edges(MultiDiEdgeQuery const &) const override final;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IOpenMultiDiGraphView);

} // namespace FlexFlow

#endif
