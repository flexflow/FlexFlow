#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_MULTIDIGRAPH_MULTIDIGRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_MULTIDIGRAPH_MULTIDIGRAPH_H

#include "utils/graph/multidigraph/multidigraph_view.h"
#include "utils/graph/multidigraph/i_multidigraph.h"

namespace FlexFlow {

struct MultiDiGraph : virtual MultiDiGraphView {
public:
  using Edge = MultiDiEdge;
  using EdgeQuery = MultiDiEdgeQuery;

  MultiDiGraph() = delete;
  MultiDiGraph(MultiDiGraph const &) = default;
  MultiDiGraph &operator=(MultiDiGraph const &) = default;

  Node add_node();
  void add_node_unsafe(Node const &);
  void remove_node_unsafe(Node const &);

  void add_edge(Edge const &e);
  void remove_edge(Edge const &e);

  std::unordered_set<Node> query_nodes(NodeQuery const &) const;
  std::unordered_set<Edge> query_edges(EdgeQuery const &) const;

  template <typename T>
  static typename std::enable_if<std::is_base_of<IMultiDiGraph, T>::value,
                                 MultiDiGraph>::type
      create() {
    return MultiDiGraph(make_cow_ptr<T>());
  }

private:
  using MultiDiGraphView::MultiDiGraphView;

  IMultiDiGraph const &get_ptr() const;
  IMultiDiGraph &get_ptr();

  friend struct GraphInternal;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(MultiDiGraph);

} // namespace FlexFlow

#endif
