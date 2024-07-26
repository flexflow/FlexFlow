#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_MULTIDIGRAPH_MULTIDIGRAPH_VIEW_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_MULTIDIGRAPH_MULTIDIGRAPH_VIEW_H

#include "utils/graph/digraph/digraph_view.h"
#include "utils/graph/multidigraph/i_multidigraph_view.h"

namespace FlexFlow {

struct MultiDiGraphView : virtual public DiGraphView {
  MultiDiGraphView(MultiDiGraphView const &) = default;
  MultiDiGraphView &operator=(MultiDiGraphView const &) = default;

  std::unordered_set<Node> query_nodes(NodeQuery const &) const;
  std::unordered_set<MultiDiEdge> query_edges(MultiDiEdgeQuery const &) const;
  Node get_multidiedge_src(MultiDiEdge const &) const;
  Node get_multidiedge_dst(MultiDiEdge const &) const;

  template <typename T, typename... Args>
  static typename std::enable_if<std::is_base_of<IMultiDiGraphView, T>::value,
                                 MultiDiGraphView>::type
      create(Args &&...args) {
    return MultiDiGraphView(make_cow_ptr<T>(std::forward<Args>(args)...));
  }

protected:
  using DiGraphView::DiGraphView;

private:
  IMultiDiGraphView const &get_interface() const;
};

} // namespace FlexFlow

#endif
