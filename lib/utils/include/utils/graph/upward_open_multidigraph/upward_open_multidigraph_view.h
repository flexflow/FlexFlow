#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_UPWARD_OPEN_MULTIDIGRAPH_UPWARD_OPEN_MULTIDIGRAPH_VIEW_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_UPWARD_OPEN_MULTIDIGRAPH_UPWARD_OPEN_MULTIDIGRAPH_VIEW_H

#include "utils/graph/multidigraph/multidigraph_view.h"
#include "utils/graph/upward_open_multidigraph/i_upward_open_multidigraph_view.h"
#include "utils/graph/upward_open_multidigraph/upward_open_multi_di_edge.dtg.h"
#include "utils/graph/upward_open_multidigraph/upward_open_multi_di_edge_query.dtg.h"

namespace FlexFlow {

struct UpwardOpenMultiDiGraphView : virtual MultiDiGraphView {
public:
  using Edge = UpwardOpenMultiDiEdge;
  using EdgeQuery = UpwardOpenMultiDiEdgeQuery;

  UpwardOpenMultiDiGraphView(UpwardOpenMultiDiGraphView const &) = default;
  UpwardOpenMultiDiGraphView &
      operator=(UpwardOpenMultiDiGraphView const &) = default;

  std::unordered_set<Node> query_nodes(NodeQuery const &);
  std::unordered_set<Edge> query_edges(EdgeQuery const &);

  template <typename T, typename... Args>
  static typename std::enable_if<
      std::is_base_of<IUpwardOpenMultiDiGraphView, T>::value,
      UpwardOpenMultiDiGraphView>::type
      create(Args &&...args) {
    return UpwardOpenMultiDiGraphView(
        cow_ptr_t<T>(std::forward<Args>(args)...));
  }

private:
  using MultiDiGraphView::MultiDiGraphView;

  IUpwardOpenMultiDiGraphView const &get_ptr() const;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(UpwardOpenMultiDiGraphView);


} // namespace FlexFlow

#endif
