#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OUTPUT_LABELLED_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OUTPUT_LABELLED_H

#include "node_labelled.h"
#include "output_labelled_interfaces.h"

namespace FlexFlow {

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
    return this->get_ptr().at(n);
  }

  OutputLabel const &at(MultiDiOutput const &o) const {
    return this->get_ptr().at(o);
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const {
    return this->get_ptr().query_nodes(q);
  }

  std::unordered_set<MultiDiEdge> query_edges(MultiDiEdgeQuery const &q) const {
    return this->get_ptr().query_edges(q);
  }

  template <typename BaseImpl, typename... Args>
  static typename std::enable_if<std::is_base_of<Interface, BaseImpl>::value,
                                 OutputLabelledMultiDiGraphView>::type
      create(Args &&...args) {
    return OutputLabelledMultiDiGraphView(
        make_cow_ptr<BaseImpl>(std::forward<Args>(args)...));
  }

protected:
  using NodeLabelledMultiDiGraphView<NodeLabel>::NodeLabelledMultiDiGraphView;

private:
  Interface const &get_ptr() const {
    return *std::reinterpret_pointer_cast<Interface const>(
        GraphView::ptr.get());
  }
};

template <typename NodeLabel, typename OutputLabel>
struct OutputLabelledMultiDiGraph
    : virtual OutputLabelledMultiDiGraphView<NodeLabel, OutputLabel> {
private:
  using Interface = IOutputLabelledMultiDiGraph<NodeLabel, OutputLabel>;

public:
  OutputLabelledMultiDiGraph(OutputLabelledMultiDiGraph const &other) = default;
  OutputLabelledMultiDiGraph &
      operator=(OutputLabelledMultiDiGraph const &other) = default;

  Node add_node(NodeLabel const &l) {
    return this->get_ptr().add_node(l);
  }

  NodePort add_node_port() {
    return this->get_ptr().add_node_port();
  }

  NodeLabel &at(Node const &n) {
    return this->get_ptr().at(n);
  }

  NodeLabel const &at(Node const &n) const {
    return this->get_ptr().at(n);
  }

  void add_output(MultiDiOutput const &o, OutputLabel const &l) {
    this->get_ptr().add_output(o, l);
  };

  void add_edge(MultiDiOutput const &o, MultiDiInput const &i) {
    this->get_ptr().add_edge(o, i);
  };

  void add_edge(MultiDiEdge const &e) {
    this->get_ptr().add_edge(e);
  }

  OutputLabel &at(MultiDiOutput const &o) {
    return this->get_ptr().at(o);
  }

  OutputLabel const &at(MultiDiOutput const &o) const {
    return this->get_ptr().at(o);
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const {
    return this->get_ptr().query_nodes(q);
  }

  std::unordered_set<MultiDiEdge> query_edges(MultiDiEdgeQuery const &q) const {
    return this->get_ptr().query_edges(q);
  }

  template <typename BaseImpl>
  static typename std::enable_if<std::is_base_of<Interface, BaseImpl>::value,
                                 OutputLabelledMultiDiGraph>::type
      create() {
    return OutputLabelledMultiDiGraph(make_cow_ptr<BaseImpl>());
  }

private:
  OutputLabelledMultiDiGraph(cow_ptr_t<Interface> ptr) : GraphView(ptr) {}

private:
  Interface &get_ptr() {
    return *std::reinterpret_pointer_cast<Interface>(
        GraphView::ptr.get_mutable());
  }

  Interface const &get_ptr() const {
    return *std::reinterpret_pointer_cast<Interface const>(
        GraphView::ptr.get());
  }
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
