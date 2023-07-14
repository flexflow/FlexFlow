#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_STANDARD_LABELLED_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_STANDARD_LABELLED_H

#include "standard_labelled_interfaces.h"
#include "node_labelled.h"

namespace FlexFlow {

template <typename NodeLabel, typename EdgeLabel>
struct LabelledMultiDiGraphView {
private:
  using Interface = ILabelledMultiDiGraphView<NodeLabel, EdgeLabel>;
public:
  LabelledMultiDiGraphView() = delete;
  LabelledMultiDiGraphView(LabelledMultiDiGraphView const &) = default;
  LabelledMultiDiGraphView &operator=(LabelledMultiDiGraphView const &) = default;

  operator NodeLabelledMultiDiGraphView<NodeLabel>() const;

  NodeLabel const &at(Node const &n) const {
    return this->ptr->at(n);
  }

  EdgeLabel const &at(MultiDiEdge const &e) const {
    return this->ptr->at(e);
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const {
    return this->ptr->query_nodes(q);
  }

  std::unordered_set<MultiDiEdge> query_edges(MultiDiEdgeQuery const &q) const {
    return this->ptr->query_edges(q);
  }

  template <typename BaseImpl, typename... Args>
  static typename std::enable_if<std::is_base_of<Interface, BaseImpl>::value,
                                 LabelledMultiDiGraphView>::type
      create(Args &&...args) {
    return LabelledMultiDiGraphView(std::make_shared<BaseImpl>(std::forward<Args>(args)...));
  }
private:
  std::shared_ptr<Interface const> ptr;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(LabelledMultiDiGraphView<int, int>);

template <typename NodeLabel, typename EdgeLabel>
struct LabelledMultiDiGraph {
private:
  using Interface = ILabelledMultiDiGraph<NodeLabel, EdgeLabel>;

public:
  LabelledMultiDiGraph() = delete;
  LabelledMultiDiGraph(LabelledMultiDiGraph const &other)
      : ptr(other.ptr->clone()) {}
  LabelledMultiDiGraph &operator=(LabelledMultiDiGraph other) {
    swap(*this, other);
    return *this;
  }

  operator LabelledMultiDiGraphView<NodeLabel, EdgeLabel>() const;

  friend void swap(LabelledMultiDiGraph &lhs, LabelledMultiDiGraph &rhs) {
    using std::swap;

    swap(lhs.ptr, rhs.ptr);
  }

  operator MultiDiGraphView() const;

  Node add_node(NodeLabel const &l) {
    return this->ptr->add_node(l);
  }

  NodeLabel &at(Node const &n) {
    return this->ptr.get_mutable()->at(n);
  }

  NodeLabel const &at(Node const &n) const {
    return this->ptr->at(n);
  }

  void add_edge(MultiDiEdge const &e, EdgeLabel const &l) {
    return this->ptr->add_edge(e, l);
  }
  EdgeLabel &at(MultiDiEdge const &e) {
    return this->ptr->at(e);
  }
  EdgeLabel const &at(MultiDiEdge const &e) const {
    return this->ptr->at(e);
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const {
    return this->ptr->query_nodes(q);
  }
  std::unordered_set<MultiDiEdge> query_edges(MultiDiEdgeQuery const &q) const {
    return this->ptr->query_edges(q);
  }

  template <typename BaseImpl>
  static typename std::enable_if<std::is_base_of<Interface, BaseImpl>::value,
                                 LabelledMultiDiGraph>::type
      create() {
    return LabelledMultiDiGraph(make_unique<BaseImpl>());
  }

private:
  LabelledMultiDiGraph(std::unique_ptr<Interface> ptr) : ptr(std::move(ptr)) {}

private:
  cow_ptr_t<Interface> ptr;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(LabelledMultiDiGraph<int, int>);

}

#endif
