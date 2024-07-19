#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_DIGRAPH_VIEW_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_DIGRAPH_VIEW_H

#include "utils/graph/digraph/directed_edge.dtg.h"
#include "utils/graph/digraph/directed_edge_query.dtg.h"
#include "utils/graph/digraph/i_digraph_view.h"
#include "utils/graph/node/graph_view.h"

namespace FlexFlow {

struct DiGraphView : virtual public GraphView {
public:
  using Edge = DirectedEdge;
  using EdgeQuery = DirectedEdgeQuery;

  DiGraphView(DiGraphView const &) = default;
  DiGraphView &operator=(DiGraphView const &) = default;

  std::unordered_set<Node> query_nodes(NodeQuery const &) const;
  std::unordered_set<Edge> query_edges(EdgeQuery const &) const;

  template <typename T, typename... Args>
  static typename std::enable_if<std::is_base_of<IDiGraphView, T>::value,
                                 DiGraphView>::type
      create(Args &&...args) {
    return DiGraphView(make_cow_ptr<T>(std::forward<Args>(args)...));
  }

protected:
  using GraphView::GraphView;

private:
  IDiGraphView const &get_ptr() const;

  friend struct GraphInternal;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(DiGraphView);

} // namespace FlexFlow

#endif
