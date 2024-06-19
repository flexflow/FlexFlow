#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_UNDIRECTED_I_UNDIRECTED_GRAPH_VIEW_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_UNDIRECTED_I_UNDIRECTED_GRAPH_VIEW_H

#include "utils/graph/node/i_graph_view.h"
#include "utils/graph/undirected/undirected_edge.h"
#include "utils/graph/undirected/undirected_edge_query.dtg.h"

namespace FlexFlow {

struct IUndirectedGraphView : public IGraphView {
  using Edge = UndirectedEdge;
  using EdgeQuery = UndirectedEdgeQuery;

  IUndirectedGraphView(IUndirectedGraphView const &) = delete;
  IUndirectedGraphView &operator=(IUndirectedGraphView const &) = delete;

  virtual std::unordered_set<Edge>
      query_edges(UndirectedEdgeQuery const &) const = 0;
  virtual ~IUndirectedGraphView() = default;

  IUndirectedGraphView *clone() const override = 0;

protected:
  IUndirectedGraphView() = default;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IUndirectedGraphView);

} // namespace FlexFlow

#endif
