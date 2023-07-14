#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_NODE_LABELLED_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_NODE_LABELLED_H

#include "node_labelled_interfaces.h"

namespace FlexFlow {

template <typename NodeLabel>
struct NodeLabelledMultiDiGraphView {
private:
  using Interface = INodeLabelledMultiDiGraphView<int>;
public:
  NodeLabelledMultiDiGraphView() = delete;
  NodeLabelledMultiDiGraphView(NodeLabelledMultiDiGraphView const &) = default;
  NodeLabelledMultiDiGraphView &operator=(NodeLabelledMultiDiGraphView const &) = default;

  NodeLabel const &at(Node const &n) const {
    return this->ptr->at(n);
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const {
    return this->ptr->query_nodes(q);
  }

  std::unordered_set<MultiDiEdge> query_edges(MultiDiEdgeQuery const &q) const {
    return this->ptr->query_edges(q);
  }

  template <typename BaseImpl, typename... Args>
  static typename std::enable_if<std::is_base_of<Interface, BaseImpl>::value,
                                 NodeLabelledMultiDiGraphView>::type
      create(Args &&...args) {
    return NodeLabelledMultiDiGraphView(std::make_shared<BaseImpl>(std::forward<Args>(args)...));
  }
private:
  std::shared_ptr<Interface const> ptr;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(NodeLabelledMultiDiGraphView<int>);

template <typename NodeLabel>
struct NodeLabelledMultiDiGraph {
private:
  using Interface = INodeLabelledMultiDiGraph<NodeLabel>;

public:
  NodeLabelledMultiDiGraph() = delete;
  NodeLabelledMultiDiGraph(NodeLabelledMultiDiGraph const &) = default;
  NodeLabelledMultiDiGraph &operator=(NodeLabelledMultiDiGraph const &) = default;

  friend void swap(NodeLabelledMultiDiGraph &lhs,
                   NodeLabelledMultiDiGraph &rhs) {
    using std::swap;

    swap(lhs.ptr, rhs.ptr);
  }

  Node add_node(NodeLabel const &l) {
    return this->ptr->add_node(l);
  }
  NodeLabel &at(Node const &n) {
    return this->ptr->at(n);
  }
  NodeLabel const &at(Node const &n) const {
    return this->ptr->at(n);
  }

  void add_edge(MultiDiEdge const &e) {
    return this->ptr->add_edge(e);
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const {
    return this->ptr->query_nodes(q);
  }
  std::unordered_set<MultiDiEdge> query_edges(MultiDiEdgeQuery const &q) const {
    return this->ptr->query_edges(q);
  }

  template <typename BaseImpl>
  static typename std::enable_if<std::is_base_of<Interface, BaseImpl>::value,
                                 NodeLabelledMultiDiGraph>::type
      create() {
    return NodeLabelledMultiDiGraph(make_unique<BaseImpl>());
  }

private:
  NodeLabelledMultiDiGraph(std::unique_ptr<Interface> ptr)
      : ptr(std::move(ptr)) {}

private:
  cow_ptr_t<Interface> ptr;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(NodeLabelledMultiDiGraph<int>);

}

#endif
