#ifndef _FLEXFLOW_UTILS_GRAPH_MULTIDIGRAPH_H
#define _FLEXFLOW_UTILS_GRAPH_MULTIDIGRAPH_H

#include "cow_ptr_t.h"
#include "multidigraph_interfaces.h"
#include "node.h"

namespace FlexFlow {
struct MultiDiGraphView {
public:
  using Edge = MultiDiEdge;
  using EdgeQuery = MultiDiEdgeQuery;

  operator GraphView() const {
    return GraphView(ptr);
  }

  friend void swap(MultiDiGraphView &, MultiDiGraphView &);

  friend void swap(MultiDiGraphView &, MultiDiGraphView &);

  IMultiDiGraphView const *unsafe() const {
    return this->ptr.get();
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &) const;
  std::unordered_set<Edge> query_edges(EdgeQuery const &query) const {
    return ptr->query_edges(query);
  }

  template <typename T, typename... Args>
  static typename std::enable_if<std::is_base_of<IMultiDiGraphView, T>::value,
                                 MultiDiGraphView>::type
      create(Args &&...args) {
    return MultiDiGraphView(
        std::make_shared<T const>(std::forward<Args>(args)...));
  }

private:
  MultiDiGraphView(std::shared_ptr<IMultiDiGraphView const> ptr)
      : ptr(ptr) {}


  friend struct MultiDiGraph;
  friend MultiDiGraphView unsafe(IMultiDiGraphView const &);

private:
  std::shared_ptr<IMultiDiGraphView const> ptr;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(MultiDiGraphView);

struct MultiDiGraph {
public:
  using Edge = MultiDiEdge;
  using EdgeQuery = MultiDiEdgeQuery;

  MultiDiGraph() = delete;
  MultiDiGraph(MultiDiGraph const &) = default;
  MultiDiGraph &operator=(MultiDiGraph const &) = default;

  operator MultiDiGraphView() const;

  friend void swap(MultiDiGraph &, MultiDiGraph &);

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
    return MultiDiGraph(make_unique<T>());
  }

private:
  MultiDiGraph(std::unique_ptr<IMultiDiGraph>);

private:
  cow_ptr_t<IMultiDiGraph> ptr;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(MultiDiGraph);

} // namespace FlexFlow

#endif
