#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_UNDIRECTED_GRAPH_VIEW_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_UNDIRECTED_GRAPH_VIEW_H

#include "utils/graph/cow_ptr_t.h"
#include "utils/graph/node/i_graph_view.h"
#include "utils/graph/node/node_query.dtg.h"

namespace FlexFlow {

struct GraphView {
  std::unordered_set<Node> query_nodes(NodeQuery const &) const;
  friend bool is_ptr_equal(GraphView const &, GraphView const &);

  template <typename T, typename... Args>
  static typename std::enable_if<std::is_base_of<IGraphView, T>::value,
                                 GraphView>::type
      create(Args &&...args) {
    return GraphView(make_cow_ptr<T>(std::forward<Args>(args)...));
  }

protected:
  GraphView();
  cow_ptr_t<IGraphView> ptr;
  GraphView(cow_ptr_t<IGraphView> ptr);

  friend struct GraphInternal;
};

} // namespace FlexFlow

#endif
