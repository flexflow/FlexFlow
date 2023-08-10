#ifndef _FLEXFLOW_UTILS_GRAPH_DIGRAPH_H
#define _FLEXFLOW_UTILS_GRAPH_DIGRAPH_H

#include "cow_ptr_t.h"
#include "node.h"
#include "utils/optional.h"
#include "utils/unique.h"
#include "utils/visitable.h"
#include <unordered_set>
#include "digraph_interfaces.h"
#include "directed_edge.h"

namespace FlexFlow {

struct DiGraphView {
public:
  using Edge = DirectedEdge;
  using EdgeQuery = DirectedEdgeQuery;

  DiGraphView() = delete;

  operator GraphView() const;

  friend void swap(DiGraphView &, DiGraphView &);

  bool operator==(DiGraphView const &) const;
  bool operator!=(DiGraphView const &) const;

  std::unordered_set<Node> query_nodes(NodeQuery const &) const;
  std::unordered_set<Edge> query_edges(EdgeQuery const &) const;

  IDiGraphView const *unsafe() const {
    return this->ptr.get();
  }

  template <typename T, typename... Args>
  static typename std::enable_if<std::is_base_of<IDiGraphView, T>::value,
                                 DiGraphView>::type
      create(Args &&...args) {
    return DiGraphView(std::make_shared<T>(std::forward<Args>(args)...));
  }

private:
  DiGraphView(std::shared_ptr<IDiGraphView const>);

  friend DiGraphView unsafe(IDiGraphView const &);

private:
  std::shared_ptr<IDiGraphView const> ptr;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(DiGraphView);

DiGraphView unsafe(IDiGraphView const &);

struct DiGraph {
public:
  using Edge = DirectedEdge;
  using EdgeQuery = DirectedEdgeQuery;

  DiGraph() = delete;
  DiGraph(DiGraph const &) = default;
  DiGraph &operator=(DiGraph const &) = default;

  operator DiGraphView() const;

  friend void swap(DiGraph &, DiGraph &);

  Node add_node() {
    Node n = Node::generate_new();
    this->ptr.get_mutable()->add_node(n);
    return n;
  }
  void remove_node(Node const &n) {
    this->ptr.get_mutable()->remove_node(n);
  }

  void add_edge(Edge const &);
  void remove_edge(Edge const &);

  std::unordered_set<Node> query_nodes(NodeQuery const &) const;
  std::unordered_set<Edge> query_edges(EdgeQuery const &) const;

  template <typename T>
  static typename std::enable_if<std::is_base_of<IDiGraph, T>::value,
                                 DiGraph>::type
      create() {
    return DiGraph(make_unique<T>());
  }

private:
  DiGraph(std::unique_ptr<IDiGraph>);

private:
  cow_ptr_t<IDiGraph> ptr;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(DiGraph);

} // namespace FlexFlow

#endif
