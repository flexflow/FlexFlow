#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_UNDIRECTED_UNDIRECTED_GRAPH_VIEW_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_UNDIRECTED_UNDIRECTED_GRAPH_VIEW_H

#include "utils/graph/node/graph_view.h"
#include "utils/graph/undirected/i_undirected_graph_view.h"
#include "utils/graph/undirected/undirected_edge.h"

namespace FlexFlow {

struct UndirectedGraphView : virtual GraphView {
public:
  using Edge = UndirectedEdge;
  using EdgeQuery = UndirectedEdgeQuery;

  UndirectedGraphView() = delete;
  UndirectedGraphView(UndirectedGraphView const &) = default;
  UndirectedGraphView &operator=(UndirectedGraphView const &) = default;

  std::unordered_set<Node> query_nodes(NodeQuery const &) const;
  std::unordered_set<Edge> query_edges(EdgeQuery const &query) const;

  template <typename T, typename... Args>
  static
      typename std::enable_if<std::is_base_of<IUndirectedGraphView, T>::value,
                              UndirectedGraphView>::type
      create(Args &&...args) {
    return UndirectedGraphView(make_cow_ptr<T>(std::forward<Args>(args)...));
  }

  using GraphView::GraphView;

  friend struct GraphInternal;

private:
  IUndirectedGraphView const &get_ptr() const;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(UndirectedGraphView);

} // namespace FlexFlow

#endif
