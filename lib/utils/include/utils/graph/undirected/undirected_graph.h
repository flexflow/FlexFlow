#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_UNDIRECTED_UNDIRECTED_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_UNDIRECTED_UNDIRECTED_GRAPH_H

#include "utils/graph/undirected/i_undirected_graph.h"
#include "utils/graph/undirected/undirected_graph_view.h"

namespace FlexFlow {

struct UndirectedGraph : virtual UndirectedGraphView {
public:
  using Edge = UndirectedEdge;
  using EdgeQuery = UndirectedEdgeQuery;

  UndirectedGraph() = delete;
  UndirectedGraph(UndirectedGraph const &) = default;
  UndirectedGraph &operator=(UndirectedGraph const &) = default;

  Node add_node();
  void add_node_unsafe(Node const &);
  void remove_node_unsafe(Node const &);

  void add_edge(Edge const &);
  void remove_edge(Edge const &);

  std::unordered_set<Node> query_nodes(NodeQuery const &) const;
  std::unordered_set<Edge> query_edges(EdgeQuery const &) const;

  template <typename T>
  static typename std::enable_if<std::is_base_of<IUndirectedGraph, T>::value,
                                 UndirectedGraph>::type
      create() {
    return UndirectedGraph(make_cow_ptr<T>());
  }

  using UndirectedGraphView::UndirectedGraphView;

private:
  IUndirectedGraph const &get_ptr() const;
  IUndirectedGraph &get_ptr();
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(UndirectedGraph);

} // namespace FlexFlow

#endif
