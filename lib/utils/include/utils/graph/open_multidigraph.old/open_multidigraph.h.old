#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_MULTIDIGRAPH_OPEN_MULTIDIGRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_MULTIDIGRAPH_OPEN_MULTIDIGRAPH_H

#include "utils/graph/open_multidigraph/open_multidigraph_view.h"
#include "utils/graph/open_multidigraph/i_open_multidigraph.h"

namespace FlexFlow {

struct OpenMultiDiGraph : virtual OpenMultiDiGraphView {
public:
  using Edge = OpenMultiDiEdge;
  using EdgeQuery = OpenMultiDiEdgeQuery;

  OpenMultiDiGraph() = delete;
  OpenMultiDiGraph(OpenMultiDiGraph const &) = default;

  Node add_node();
  void add_node_unsafe(Node const &);
  void remove_node_unsafe(Node const &);

  void add_edge(Edge const &);
  void remove_edge(Edge const &);

  std::unordered_set<Edge> query_edges(EdgeQuery const &) const;

  template <typename T>
  static typename std::enable_if<std::is_base_of<IOpenMultiDiGraph, T>::value,
                                 OpenMultiDiGraph>::type
      create() {
    return OpenMultiDiGraph(make_cow_ptr<T>());
  }

private:
  using OpenMultiDiGraphView::OpenMultiDiGraphView;

  IOpenMultiDiGraph const &get_ptr() const;
  IOpenMultiDiGraph &get_ptr();

  friend struct GraphInternal;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(OpenMultiDiGraph);


} // namespace FlexFlow

#endif
