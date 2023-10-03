#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_STANDARD_LABELLED_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_STANDARD_LABELLED_H

#include "node_labelled.h"

namespace FlexFlow {

template <typename NodeLabel, typename EdgeLabel>
struct ILabelledMultiDiGraphView
    : public INodeLabelledMultiDiGraphView<NodeLabel> {
  ILabelledMultiDiGraphView() = default;
  ILabelledMultiDiGraphView(ILabelledMultiDiGraphView const &) = delete;
  ILabelledMultiDiGraphView &
      operator=(ILabelledMultiDiGraphView const &) = delete;

  virtual ~ILabelledMultiDiGraphView() = default;

  virtual EdgeLabel const &at(MultiDiEdge const &) const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(ILabelledMultiDiGraphView<int, int>);

template <typename NodeLabel, typename EdgeLabel>
struct LabelledMultiDiGraphView : virtual public NodeLabelledMultiDiGraphView<NodeLabel> {
private:
  using Interface = ILabelledMultiDiGraphView<NodeLabel, EdgeLabel>;

public:
  LabelledMultiDiGraphView() = delete;
  LabelledMultiDiGraphView(LabelledMultiDiGraphView const &) = default;
  LabelledMultiDiGraphView &
      operator=(LabelledMultiDiGraphView const &) = default;

  NodeLabel const &at(Node const &n) const {
    return get_ptr()->at(n);
  }

  EdgeLabel const &at(MultiDiEdge const &e) const {
    return get_ptr()->at(e);
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const {
    return get_ptr()->query_nodes(q);
  }

  std::unordered_set<MultiDiEdge> query_edges(MultiDiEdgeQuery const &q) const {
    return get_ptr()->query_edges(q);
  }

  template <typename BaseImpl, typename... Args>
  static typename std::enable_if<std::is_base_of<Interface, BaseImpl>::value,
                                 LabelledMultiDiGraphView>::type
      create(Args &&...args) {
    return LabelledMultiDiGraphView(
        make_cow_ptr<BaseImpl>(std::forward<Args>(args)...));
  }

protected:
  LabelledMultiDiGraphView(cow_ptr_t<Interface const> ptr) : NodeLabelledMultiDiGraphView<NodeLabel>(ptr) {}
  cow_ptr_t<Interface const> get_ptr() const {
    return static_assert<cow_ptr_t<Interface const>>(ptr);
  }
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(LabelledMultiDiGraphView<int, int>);

template <typename NodeLabel, typename EdgeLabel>
struct LabelledMultiDiGraph : virtual LabelledMultiDiGraphView<NodeLabel, EdgeLabel> {
private:
  using Interface = ILabelledMultiDiGraph<NodeLabel, EdgeLabel>;
  using INodeLabel = ILabel<Node, NodeLabel>;
  using IEdgeLabel = ILabel<MultiDiEdge, EdgeLabel>;

public:
  LabelledMultiDiGraph() = delete;
  LabelledMultiDiGraph(LabelledMultiDiGraph const &other) = default;
  LabelledMultiDiGraph &operator=(LabelledMultiDiGraph other) = default;

  friend void swap(LabelledMultiDiGraph &lhs, LabelledMultiDiGraph &rhs) {
    using std::swap;

    swap(lhs.ptr, rhs.ptr);
  }

  Node add_node(NodeLabel const &l) {
    Node n = MultiDiGraph::add_node();
    nl->add_label(n, l);
    return n;
  }

  NodePort add_node_port() {
    return this->get_ptr()->add_node_port();
  }

  NodeLabel &at(Node const &n) {
    return nl->get_label(n);
  }

  NodeLabel const &at(Node const &n) const {
    return nl->get_label(n);
  }

  void add_edge(MultiDiEdge const &e, EdgeLabel const &l) {
    return this->get_ptr()->add_edge(e, l);
  }
  EdgeLabel &at(MultiDiEdge const &e) {
    return el->get_label(e);
  }
  EdgeLabel const &at(MultiDiEdge const &e) const {
    return el->get_label(e);
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const {
    return this->get_ptr()->query_nodes(q);
  }
  std::unordered_set<MultiDiEdge> query_edges(MultiDiEdgeQuery const &q) const {
    return this->get_ptr()->query_edges(q);
  }

  template <typename BaseImpl, typename N, typename E>
  static typename std::enable_if<std::conjunction<is_base_of<Interface, BaseImpl>,
                                                  is_base_of<INodeLabel, N>,
                                                  is_base_of<IEdgeLabel, E>>::value,
                                 LabelledMultiDiGraph>::type
      create() {
    return LabelledMultiDiGraph(make_cow_ptr<BaseImpl>(),
                                make_cow_ptr<N>(),
                                make_cow_ptr<E>());
  }

private:
  LabelledMultiDiGraph(cow_ptr_t<Interface> ptr,
                       cow_ptr_t<INodeLabel> nl,
                       cow_ptr_t<IEdgeLabel> el) : LabelledMultiDiGraphView(ptr), nl(nl), el(el) {}

  cow_ptr_t<Interface> get_ptr() const {
    return static_cast<cow_ptr_t<Interface>>(ptr);
  }

  cow_ptr_t<INodeLabel> nl;
  cow_ptr_t<IEdgeLabel> el;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(LabelledMultiDiGraph<int, int>);

} // namespace FlexFlow

#endif
