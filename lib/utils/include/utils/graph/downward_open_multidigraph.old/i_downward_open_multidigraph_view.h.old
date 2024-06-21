#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DOWNWARD_OPEN_MULTIDIGRAPH_I_DOWNWARD_OPEN_MULTIDIGRAPH_VIEW_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DOWNWARD_OPEN_MULTIDIGRAPH_I_DOWNWARD_OPEN_MULTIDIGRAPH_VIEW_H

#include "utils/graph/open_multidigraph/i_open_multidigraph_view.h"
#include "utils/graph/downward_open_multidigraph/downward_open_multi_di_edge.dtg.h"
#include "utils/graph/downward_open_multidigraph/downward_open_multi_di_edge_query.dtg.h"
#include "utils/graph/open_multidigraph/open_multi_di_edge.dtg.h"
#include "utils/graph/open_multidigraph/open_multi_di_edge_query.dtg.h"

namespace FlexFlow {

struct IDownwardOpenMultiDiGraphView : virtual public IOpenMultiDiGraphView {
  virtual std::unordered_set<DownwardOpenMultiDiEdge>
      query_edges(DownwardOpenMultiDiEdgeQuery const &) const = 0;

  std::unordered_set<OpenMultiDiEdge>
      query_edges(OpenMultiDiEdgeQuery const &q) const final;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IDownwardOpenMultiDiGraphView);

} // namespace FlexFlow

#endif
