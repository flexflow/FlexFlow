#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_I_DIGRAPH_VIEW_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_I_DIGRAPH_VIEW_H

#include "utils/graph/digraph/directed_edge.dtg.h"
#include "utils/graph/digraph/directed_edge_query.dtg.h"
#include "utils/graph/node/i_graph_view.h"
#include "utils/graph/node/node.dtg.h"

namespace FlexFlow {

struct IDiGraphView : virtual public IGraphView {
public:
  using Edge = DirectedEdge;
  using EdgeQuery = DirectedEdgeQuery;

  IDiGraphView() = default;

  IDiGraphView(IDiGraphView const &) = delete;
  IDiGraphView &operator=(IDiGraphView const &) = delete;

  virtual std::unordered_set<Edge> query_edges(EdgeQuery const &) const = 0;
  virtual ~IDiGraphView() = default;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IDiGraphView);

} // namespace FlexFlow

#endif
