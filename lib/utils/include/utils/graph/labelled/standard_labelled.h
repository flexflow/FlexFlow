#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_STANDARD_LABELLED_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_STANDARD_LABELLED_H

#include "node_labelled.h"
#include "utils/graph/labelled/standard_labelled_interfaces.h"

namespace FlexFlow {

// template <typename NodeLabel, typename EdgeLabel>
// struct ILabelledMultiDiGraphView
//     : public INodeLabelledMultiDiGraphView<NodeLabel> {
//   ILabelledMultiDiGraphView() = default;
//   ILabelledMultiDiGraphView(ILabelledMultiDiGraphView const &) = delete;
//   ILabelledMultiDiGraphView &
//       operator=(ILabelledMultiDiGraphView const &) = delete;

//   virtual ~ILabelledMultiDiGraphView() = default;

//   virtual EdgeLabel const &at(MultiDiEdge const &) const = 0;
// };
// CHECK_RC_COPY_VIRTUAL_COMPLIANT(ILabelledMultiDiGraphView<int, int>);

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
  LabelledMultiDiGraphView(cow_ptr_t<Interface const> ptr)
      : NodeLabelledMultiDiGraphView<NodeLabel>(ptr) {
  } // todo: this may have some problem, because it seems we don't have
    // constructor method NodeLabelledMultiDiGraphView<NodeLabel>(ptr
  cow_ptr_t<Interface> get_ptr() const {
    return cow_ptr_t(static_cast<Interface const &>(*GraphView::ptr));
  }
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(LabelledMultiDiGraphView<int, int>);

template <typename NodeLabel, typename EdgeLabel>
struct LabelledMultiDiGraph
    : virtual LabelledMultiDiGraphView<NodeLabel, EdgeLabel> {
private:
  using Interface = IMultiDiGraph;
  using INodeLabel = ILabelling<Node, NodeLabel>;
  using IEdgeLabel = ILabelling<MultiDiEdge, EdgeLabel>;

public:
  // LabelledMultiDiGraph() = delete;
  LabelledMultiDiGraph(LabelledMultiDiGraph const &other) = default;
  LabelledMultiDiGraph &operator=(LabelledMultiDiGraph const &other) = default;

  Node add_node(NodeLabel const &l) {
    Node n = MultiDiGraph::add_node();
    nl.get_mutable()->add_label(n, l);
    return n;
  }

  NodePort add_node_port() {
    return get_ptr().add_node_port();
  }

  NodeLabel &at(Node const &n) {
    return get_nodelabel_ptr().get_label(n);
  }

  NodeLabel const &at(Node const &n) const {
    return nl->get_label(n);
  }

  void add_edge(MultiDiEdge const &e) {
    return get_ptr().add_edge(e);
  }

  void add_label(MultiDiEdge const &e, EdgeLabel const &l) {
    el.get_mutable()->add_label(e, l);
  }

  EdgeLabel &at(MultiDiEdge const &e) {
    return el->get_label(e);
  }

  EdgeLabel const &at(MultiDiEdge const &e) const {
    return get_edgelabel_ptr().get_label(e);
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const {
    return get_ptr().query_nodes(q);
  }

  std::unordered_set<MultiDiEdge> query_edges(MultiDiEdgeQuery const &q) const {
    return get_ptr().query_edges(q);
  }

  template <typename BaseImpl, typename N, typename E>
  static typename std::enable_if<
      std::conjunction<std::is_base_of<Interface, BaseImpl>,
                       std::is_base_of<INodeLabel, N>,
                       std::is_base_of<IEdgeLabel, E>>::value,
      LabelledMultiDiGraph>::type
      create() {
    return LabelledMultiDiGraph(
        make_cow_ptr<BaseImpl>(), make_cow_ptr<N>(), make_cow_ptr<E>());
  }

private:
  LabelledMultiDiGraph(cow_ptr_t<Interface> ptr,
                       cow_ptr_t<INodeLabel> nl,
                       cow_ptr_t<IEdgeLabel> el)
      : LabelledMultiDiGraphView<NodeLabel, EdgeLabel>(ptr), nl(nl), el(el) {}
  // todo: this may have some problem, because it seems we don't have
  // constructor method  LabelledMultiDiGraphView<NodeLabel, EdgeLabel>(ptr)

  Interface &get_ptr() const {
    return *std::reinterpret_pointer_cast<Interface>(
        GraphView::ptr.get_mutable());
  }

  INodeLabel &get_nodelabel_ptr() const {
    return *std::reinterpret_pointer_cast<INodeLabel>(nl.get_mutable());
  }

  IEdgeLabel &get_edgelabel_ptr() const {
    return *std::reinterpret_pointer_cast<IEdgeLabel>(el.get_mutable());
  }

  cow_ptr_t<INodeLabel> nl;
  cow_ptr_t<IEdgeLabel> el;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(LabelledMultiDiGraph<int, int>);

} // namespace FlexFlow

#endif
