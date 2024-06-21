#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_MULTIDIGRAPH_OPEN_MULTIDIGRAPH_VIEW_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_MULTIDIGRAPH_OPEN_MULTIDIGRAPH_VIEW_H

#include "utils/graph/multidigraph/multidigraph_view.h"
#include "utils/graph/open_multidigraph/i_open_multidigraph_view.h"
#include "utils/graph/open_multidigraph/open_multi_di_edge.dtg.h"
#include "utils/graph/open_multidigraph/open_multi_di_edge_query.dtg.h"

namespace FlexFlow {

struct OpenMultiDiGraphView : virtual MultiDiGraphView {
public:
  using Edge = OpenMultiDiEdge;
  using EdgeQuery = OpenMultiDiEdgeQuery;

  OpenMultiDiGraphView(OpenMultiDiGraphView const &) = default;
  OpenMultiDiGraphView &operator=(OpenMultiDiGraphView const &) = default;

  std::unordered_set<Node> query_nodes(NodeQuery const &) const;
  std::unordered_set<Edge> query_edges(EdgeQuery const &) const;

  template <typename T, typename... Args>
  static
      typename std::enable_if<std::is_base_of<IOpenMultiDiGraphView, T>::value,
                              OpenMultiDiGraphView>::type
      create(Args &&...args) {
    return OpenMultiDiGraphView(make_cow_ptr<T>(std::forward<Args>(args)...));
  }

protected:
  using MultiDiGraphView::MultiDiGraphView;

private:
  IOpenMultiDiGraphView const &get_ptr() const;

  friend struct GraphInternal;
};

} // namespace FlexFlow

#endif
