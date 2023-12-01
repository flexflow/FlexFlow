#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OUTPUT_LABELLED_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OUTPUT_LABELLED_H

#include "standard_labelled.h"

namespace FlexFlow {

template <typename NodeLabel, typename OutputLabel>
struct IOutputLabelledMultiDiGraphView
    : public INodeLabelledMultiDiGraphView<NodeLabel> {
  IOutputLabelledMultiDiGraphView() = default;
  IOutputLabelledMultiDiGraphView(IOutputLabelledMultiDiGraphView const &) =
      delete;
  IOutputLabelledMultiDiGraphView &
      operator=(IOutputLabelledMultiDiGraphView const &) = delete;

  virtual OutputLabel const &at(MultiDiOutput const &) = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IOutputLabelledMultiDiGraphView<int, int>);

template <typename NodeLabel, typename OutputLabel>
struct OutputLabelledMultiDiGraphView
    : virtual public NodeLabelledMultiDiGraphView<NodeLabel> {
private:
  using Interface = IOutputLabelledMultiDiGraphView<NodeLabel, OutputLabel>;

public:
  OutputLabelledMultiDiGraphView(OutputLabelledMultiDiGraphView const &) =
      default;
  OutputLabelledMultiDiGraphView &
      operator=(OutputLabelledMultiDiGraphView const &) = default;

  NodeLabel const &at(Node const &n) const {
    return get_ptr().at(n);
  }

  OutputLabel const &at(MultiDiOutput const &o) const {
    return get_ptr().at(o);
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const {
    return get_ptr().query_nodes(q);
  }

  std::unordered_set<MultiDiEdge> query_edges(MultiDiEdgeQuery const &q) const {
    return get_ptr().query_edges(q);
  }

  template <typename BaseImpl, typename... Args>
  static typename std::enable_if<std::is_base_of<Interface, BaseImpl>::value,
                                 OutputLabelledMultiDiGraphView>::type
      create(Args &&...args) {
    return OutputLabelledMultiDiGraphView(
        make_cow_ptr<BaseImpl>(std::forward<Args>(args)...));
  }

protected:
  OutputLabelledMultiDiGraphView(cow_ptr_t<Interface const> ptr)
      : NodeLabelledMultiDiGraphView<NodeLabel>(ptr) {}

private:
  Interface &get_ptr() const {
    return *std::reinterpret_pointer_cast<Interface>(
        GraphView::ptr.get_mutable());
  }
};

template <typename NodeLabel, typename OutputLabel>
struct OutputLabelledMultiDiGraph
    : virtual OutputLabelledMultiDiGraphView<NodeLabel, OutputLabel> {
private:
  using Interface = IMultiDiGraph;
  using INodeLabel = ILabelling<Node, NodeLabel>;
  using IOutputLabel = ILabelling<MultiDiOutput, OutputLabel>;

public:
  OutputLabelledMultiDiGraph(OutputLabelledMultiDiGraph const &other) = default;
  OutputLabelledMultiDiGraph &
      operator=(OutputLabelledMultiDiGraph const &other) = default;

  Node add_node(NodeLabel const &l) {
    Node n = get_ptr().add_node();
    nl.get_mutable()->add_label(n, l);
    return n;
  }

  NodePort add_node_port() {
    return get_ptr().add_node_port();
  }

  NodeLabel &at(Node const &n) {
    return nl.get_mutable()->get_label(n);
  }

  NodeLabel const &at(Node const &n) const {
    return nl->get_label(n);
  }

  void add_edge(MultiDiOutput const &o, OutputLabel const &l) {
    ol.get_mutable()->add_label(o, l);
  };

  void add_edge(MultiDiOutput const &o, MultiDiInput const &i) {
    return get_ptr().add_edge(o, i);
  };

  void add_edge(MultiDiEdge const &e) {
    return get_ptr().add_edge(e);
  }

  OutputLabel &at(MultiDiOutput const &o) {
    return ol.get_mutable()->get_label(o);
  }

  OutputLabel const &at(MultiDiOutput const &o) const {
    return ol->get_label(o);
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const {
    return get_ptr().query_nodes(q);
  }
  std::unordered_set<MultiDiEdge> query_edges(MultiDiEdgeQuery const &q) const {
    return get_ptr().query_edges(q);
  }

  template <typename BaseImpl, typename N, typename O>
  static typename std::enable_if<
      std::conjunction<std::is_base_of<Interface, BaseImpl>,
                       std::is_base_of<INodeLabel, N>,
                       std::is_base_of<IOutputLabel, O>>::value,
      OutputLabelledMultiDiGraph>::type
      create() {
    return OutputLabelledMultiDiGraph(
        make_cow_ptr<BaseImpl>(), make_cow_ptr<N>(), make_cow_ptr<O>());
  }

private:
  OutputLabelledMultiDiGraph(cow_ptr_t<Interface> ptr,
                             cow_ptr_t<INodeLabel> nl,
                             cow_ptr_t<IOutputLabel> ol)
      : OutputLabelledMultiDiGraphView<NodeLabel, OutputLabel>(ptr), nl(nl),
        ol(ol) {
  } // this exists some problem, interface is IMultiDiGraph, but
    // OutputLabelledMultiDiGraphView needs  IOutputLabelledMultiDiGraphView

private:
  Interface &get_ptr() const {
    return *std::reinterpret_pointer_cast<Interface>(
        GraphView::ptr.get_mutable());
  }
  cow_ptr_t<INodeLabel> nl;
  cow_ptr_t<IOutputLabel> ol;
};

template <typename NodeLabel,
          typename T,
          typename std::enable_if<
              (std::is_convertible<T, NodeLabelledMultiDiGraphView<NodeLabel>>::
                   value &&
               !std::is_same<T, T>::value),
              bool>::type = true>
NodeLabel const &at(T const &g, Node const &n) {
  return g.at(n);
}

} // namespace FlexFlow

#endif
