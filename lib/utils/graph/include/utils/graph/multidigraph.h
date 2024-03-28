#ifndef _FLEXFLOW_UTILS_GRAPH_MULTIDIGRAPH_H
#define _FLEXFLOW_UTILS_GRAPH_MULTIDIGRAPH_H

#include "utils/smart_ptrs/cow_ptr_t.h"
#include "digraph.h"
#include "multidiedge.h"
#include "multidigraph_interfaces.h"
#include "node.h"

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

struct MultiDiGraph : virtual MultiDiGraphView {
public:
  using Edge = MultiDiEdge;
  using EdgeQuery = MultiDiEdgeQuery;

  MultiDiGraph() = delete;
  MultiDiGraph(MultiDiGraph const &) = default;
  MultiDiGraph &operator=(MultiDiGraph const &) = default;

  Node add_node();
  NodePort add_node_port();
  void add_node_unsafe(Node const &);
  void add_node_port_unsafe(NodePort const &);
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
