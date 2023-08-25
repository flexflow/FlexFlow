#ifndef _FLEXFLOW_UTILS_GRAPH_DIGRAPH_H
#define _FLEXFLOW_UTILS_GRAPH_DIGRAPH_H

#include "cow_ptr_t.h"
#include "digraph_interfaces.h"
#include "node.h"
#include "utils/optional.h"
#include "utils/unique.h"
#include "utils/visitable.h"
#include <unordered_set>

namespace FlexFlow {

struct DiGraphView {
public:
  using Edge = DirectedEdge;
  using EdgeQuery = DirectedEdgeQuery;

  DiGraphView() = delete;

  operator GraphView() const;

  friend void swap(DiGraphView &, DiGraphView &);

  std::unordered_set<Node> query_nodes(NodeQuery const &) const;
  std::unordered_set<Edge> query_edges(EdgeQuery const &) const;
  friend bool is_ptr_equal(DiGraphView const &, DiGraphView const &);

  template <typename T, typename... Args>
  static typename std::enable_if<std::is_base_of<IDiGraphView, T>::value,
                                 DiGraphView>::type
      create(Args &&...args) {
    return DiGraphView(std::make_shared<T>(std::forward<Args>(args)...));
  }

  static DiGraphView
      unsafe_create_without_ownership(IDiGraphView const &graphView);

private:
  DiGraphView(std::shared_ptr<IDiGraphView const> ptr) : ptr(ptr) {}

  friend struct GraphInternal;

  std::shared_ptr<IDiGraphView const> ptr;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(DiGraphView);

struct DiGraph {
public:
  using Edge = DirectedEdge;
  using EdgeQuery = DirectedEdgeQuery;

  DiGraph() = delete;
  DiGraph(DiGraph const &) = default;
  DiGraph &operator=(DiGraph const &) = default;

  operator DiGraphView() const;

  friend void swap(DiGraph &, DiGraph &);

  Node add_node();
  void add_node_unsafe(Node const &);
  void remove_node_unsafe(Node const &);

  void add_edge(Edge const &);
  void remove_edge(Edge const &);

  std::unordered_set<Node> query_nodes(NodeQuery const &) const;
  std::unordered_set<Edge> query_edges(EdgeQuery const &) const;

  template <typename T>
  static typename std::enable_if<std::is_base_of<IDiGraph, T>::value,
                                 DiGraph>::type
      create() {
    return DiGraph(make_cow_ptr<T>());
  }

private:
  DiGraph(cow_ptr_t<IDiGraph>);

  friend struct GraphInternal;

private:
  cow_ptr_t<IDiGraph> ptr;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(DiGraph);

} // namespace FlexFlow

#endif
