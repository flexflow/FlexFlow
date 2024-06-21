#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_UPWARD_OPEN_MULTIDIGRAPH_I_UPWARD_OPEN_MULTIDIGRAPH_VIEW_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_UPWARD_OPEN_MULTIDIGRAPH_I_UPWARD_OPEN_MULTIDIGRAPH_VIEW_H

#include "utils/graph/open_multidigraph/i_open_multidigraph_view.h"
#include "utils/graph/upward_open_multidigraph/upward_open_multi_di_edge.dtg.h"
#include "utils/graph/upward_open_multidigraph/upward_open_multi_di_edge_query.dtg.h"

namespace FlexFlow {

struct IUpwardOpenMultiDiGraphView : virtual public IOpenMultiDiGraphView {
  virtual std::unordered_set<UpwardOpenMultiDiEdge>
      query_edges(UpwardOpenMultiDiEdgeQuery const &) const = 0;

  std::unordered_set<OpenMultiDiEdge>
      query_edges(OpenMultiDiEdgeQuery const &q) const final;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IUpwardOpenMultiDiGraphView);

} // namespace FlexFlow

#endif
