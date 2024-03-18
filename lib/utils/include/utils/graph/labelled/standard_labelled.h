#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_STANDARD_LABELLED_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_STANDARD_LABELLED_H

#include "node_labelled.h"
#include "standard_labelled_interfaces.h"

namespace FlexFlow {

template <typename NodeLabel, typename EdgeLabel>
struct LabelledMultiDiGraphView
    : virtual public NodeLabelledMultiDiGraphView<NodeLabel> {
private:
  using Interface = ILabelledMultiDiGraphView<NodeLabel, EdgeLabel>;

public:
  // LabelledMultiDiGraphView() = delete;
  LabelledMultiDiGraphView(LabelledMultiDiGraphView const &) = default;
  LabelledMultiDiGraphView &
      operator=(LabelledMultiDiGraphView const &) = default;

  NodeLabel const &at(Node const &n) const {
    return get_ptr().at(n);
  }

  EdgeLabel const &at(MultiDiEdge const &e) const {
    return get_ptr().at(e);
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const {
    return get_ptr().query_nodes(q);
  }

  std::unordered_set<MultiDiEdge> query_edges(MultiDiEdgeQuery const &q) const {
    return get_ptr().query_edges(q);
  }

  template <typename BaseImpl, typename... Args>
  static typename std::enable_if<std::is_base_of<Interface, BaseImpl>::value,
                                 LabelledMultiDiGraphView>::type
      create(Args &&...args) {
    return LabelledMultiDiGraphView(
        make_cow_ptr<BaseImpl>(std::forward<Args>(args)...));
  }

protected:
  LabelledMultiDiGraphView(cow_ptr_t<Interface const> ptr)
      : NodeLabelledMultiDiGraphView<NodeLabel>(ptr) {}

  Interface const &get_ptr() const {
    return *std::reinterpret_pointer_cast<Interface const>(
        GraphView::ptr.get());
  }
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(LabelledMultiDiGraphView<int, int>);

template <typename NodeLabel, typename EdgeLabel>
struct LabelledMultiDiGraph
    : virtual LabelledMultiDiGraphView<NodeLabel, EdgeLabel> {
private:
  using Interface = ILabelledMultiDiGraph<NodeLabel, EdgeLabel>;

public:
  LabelledMultiDiGraph(LabelledMultiDiGraph const &other) = default;
  LabelledMultiDiGraph &operator=(LabelledMultiDiGraph const &other) = default;

  Node add_node(NodeLabel const &l) {
    return this->get_ptr().add_node();
  }

  NodePort add_node_port() {
    return this->get_ptr().add_node_port();
  }

  NodeLabel &at(Node const &n) {
    return this->get_ptr().at(n);
  }

  void add_edge(MultiDiEdge const &e, EdgeLabel const &l) {
    return this->get_ptr().add_edge(e, l);
  }

  EdgeLabel &at(MultiDiEdge const &e) {
    return this->get_ptr().at(e);
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const {
    return this->get_ptr().query_nodes(q);
  }

  std::unordered_set<MultiDiEdge> query_edges(MultiDiEdgeQuery const &q) const {
    return this->get_ptr().query_edges(q);
  }

  using LabelledMultiDiGraphView<NodeLabel, EdgeLabel>::at;

  template <typename BaseImpl, typename N>
  static typename std::enable_if<std::is_base_of<Interface, BaseImpl>::value,
                                 LabelledMultiDiGraph>::type
      create() {
    return LabelledMultiDiGraph(make_cow_ptr<BaseImpl>());
  }

private:
  LabelledMultiDiGraph(cow_ptr_t<Interface> ptr) : GraphView(ptr) {}

  Interface &get_ptr() {
    return *std::reinterpret_pointer_cast<Interface>(
        GraphView::ptr.get_mutable());
  }

  Interface const &get_ptr() const {
    return *std::reinterpret_pointer_cast<Interface const>(
        GraphView::ptr.get());
  }
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(LabelledMultiDiGraph<int, int>);

} // namespace FlexFlow

#endif
