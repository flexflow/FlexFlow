#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DOWNWARD_OPEN_MULTIDIGRAPH_DOWNWARD_OPEN_MULTIDIGRAPH_VIEW_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DOWNWARD_OPEN_MULTIDIGRAPH_DOWNWARD_OPEN_MULTIDIGRAPH_VIEW_H

#include "utils/graph/downward_open_multidigraph/downward_open_multi_di_edge.dtg.h"
#include "utils/graph/downward_open_multidigraph/downward_open_multi_di_edge_query.dtg.h"
#include "utils/graph/downward_open_multidigraph/i_downward_open_multidigraph_view.h"
#include "utils/graph/multidigraph/multidigraph_view.h"

namespace FlexFlow {

struct DownwardOpenMultiDiGraphView : virtual MultiDiGraphView {
public:
  using Edge = DownwardOpenMultiDiEdge;
  using EdgeQuery = DownwardOpenMultiDiEdgeQuery;
  using Interface = IDownwardOpenMultiDiGraphView;

  DownwardOpenMultiDiGraphView(DownwardOpenMultiDiGraphView const &) = default;
  DownwardOpenMultiDiGraphView &
      operator=(DownwardOpenMultiDiGraphView const &) = default;

  std::unordered_set<Node> query_nodes(NodeQuery const &) const;
  std::unordered_set<Edge> query_edges(EdgeQuery const &) const;

  template <typename T, typename... Args>
  static typename std::enable_if<
      std::is_base_of<IDownwardOpenMultiDiGraphView, T>::value,
      DownwardOpenMultiDiGraphView>::type
      create(Args &&...args) {
    return DownwardOpenMultiDiGraphView(
        make_cow_ptr<T>(std::forward<Args>(args)...));
  }

private:
  using MultiDiGraphView::MultiDiGraphView;

  Interface const &get_ptr() const;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(DownwardOpenMultiDiGraphView);


} // namespace FlexFlow

#endif
