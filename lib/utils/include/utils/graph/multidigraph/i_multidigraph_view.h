#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_MULTIDIGRAPH_I_MULTIDIGRAPH_VIEW_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_MULTIDIGRAPH_I_MULTIDIGRAPH_VIEW_H

#include "utils/graph/digraph/i_digraph_view.h"
#include "utils/graph/multidigraph/multidiedge.dtg.h"
#include "utils/graph/multidigraph/multidiedge_query.dtg.h"

namespace FlexFlow {

struct IMultiDiGraphView : virtual public IDiGraphView {
public:
  using Edge = MultiDiEdge;
  using EdgeQuery = MultiDiEdgeQuery;

  IMultiDiGraphView() = default;

  IMultiDiGraphView(IMultiDiGraphView const &) = delete;
  IMultiDiGraphView &operator=(IMultiDiGraphView const &) = delete;

  virtual std::unordered_set<Edge> query_edges(EdgeQuery const &) const = 0;
  virtual Node get_multidiedge_src(MultiDiEdge const &) const = 0;
  virtual Node get_multidiedge_dst(MultiDiEdge const &) const = 0;

  virtual ~IMultiDiGraphView() = default;

  std::unordered_set<DirectedEdge>
      query_edges(DirectedEdgeQuery const &) const override final;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IMultiDiGraphView);

} // namespace FlexFlow

#endif
