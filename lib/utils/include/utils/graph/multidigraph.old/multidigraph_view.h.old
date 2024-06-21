#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_MULTIDIGRAPH_MULTIDIGRAPH_VIEW_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_MULTIDIGRAPH_MULTIDIGRAPH_VIEW_H

#include "utils/graph/digraph/digraph_view.h"
#include "utils/graph/multidigraph/multi_di_edge.dtg.h"
#include "utils/graph/multidigraph/i_multidigraph_view.h"

namespace FlexFlow {

struct MultiDiGraphView : virtual DiGraphView {
public:
  using Edge = MultiDiEdge;
  using EdgeQuery = MultiDiEdgeQuery;

  MultiDiGraphView(MultiDiGraphView const &) = default;
  MultiDiGraphView &operator=(MultiDiGraphView const &) = default;

  std::unordered_set<Node> query_nodes(NodeQuery const &) const;
  std::unordered_set<Edge> query_edges(EdgeQuery const &) const;

  template <typename T, typename... Args>
  static typename std::enable_if<std::is_base_of<IMultiDiGraphView, T>::value,
                                 MultiDiGraphView>::type
      create(Args &&...args) {
    return MultiDiGraphView(make_cow_ptr<T>(std::forward<Args>(args)...));
  }

protected:
  using DiGraphView::DiGraphView;

private:
  IMultiDiGraphView const &get_ptr() const;

  friend struct GraphInternal;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(MultiDiGraphView);

} // namespace FlexFlow

#endif
