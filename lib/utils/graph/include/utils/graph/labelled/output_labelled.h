#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OUTPUT_LABELLED_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OUTPUT_LABELLED_H

#include "output_labelled_interfaces.h"
#include "standard_labelled.h"

namespace FlexFlow {

template <typename NodeLabel, typename OutputLabel>
struct OutputLabelledMultiDiGraphView : virtual public NodeLabelledMultiDiGraphView<NodeLabel> {
private:
  using Interface = IOutputLabelledMultiDiGraphView<NodeLabel, OutputLabel>;

public:
  OutputLabelledMultiDiGraphView() = delete;
  OutputLabelledMultiDiGraphView(OutputLabelledMultiDiGraphView const &) = default;
  OutputLabelledMultiDiGraphView &operator=(OutputLabelledMultiDiGraphView const &) = default;

  NodeLabel const &at(Node const &n) const {
    return get_ptr()->at(n);
  }

  OutputLabel const &at(MultiDiOutput const &o) const {
    return get_ptr()->at(o);
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const {
    return get_ptr()->query_nodes(q);
  }

  std::unordered_set<MultiDiEdge> query_edges(MultiDiEdgeQuery const &q) const {
    return get_ptr()->query_edges(q);
  }

private:
  OutputLabelledMultiDiGraphView(std::shared_ptr<Interface const> ptr) : NodeLabelledMultiDiGraphView<NodeLabel>(ptr) {}
  std::shared_ptr<Interface const> get_ptr() const {
    return static_assert<std::shared_ptr<Interface const>>(ptr);
  }
};

template <typename NodeLabel, typename OutputLabel>
struct OutputLabelledMultiDiGraph : virtual OutputLabelledMultiDiGraphView<NodeLabel, OutputLabel> {
private:
  using Interface = IOutputLabelledMultiDiGraph<NodeLabel, OutputLabel>;
  using INodeLabel = ILabel<Node, NodeLabel>;
  using IOutputLabel = ILabel<MultiDiOutput, OutputLabel>;

public:
  OutputLabelledMultiDiGraph() = delete;
  OutputLabelledMultiDiGraph(OutputLabelledMultiDiGraph const &other) = default;
  OutputLabelledMultiDiGraph &
      operator=(OutputLabelledMultiDiGraph const &other) = default;

  friend void swap(OutputLabelledMultiDiGraph &lhs,
                   OutputLabelledMultiDiGraph &rhs) {
    using std::swap;

    swap(lhs.ptr, rhs.ptr);
  }

  Node add_node(NodeLabel const &l) {
    return this->ptr->add_node(l);
  }

  NodePort add_node_port() {
    return this->ptr->add_node_port();
  }

  NodeLabel &at(Node const &n) {
    return this->ptr->at(n);
  }

  NodeLabel const &at(Node const &n) const {
    return this->ptr->at(n);
  }

  void add_output(MultiDiOutput const &o, OutputLabel const &l) {
    return this->ptr->add_output(o, l);
  };
  void add_edge(MultiDiOutput const &o, MultiDiInput const &i) {
    return this->ptr->add_edge(o, i);
  };

  OutputLabel &at(MultiDiOutput const &o) {
    return this->ptr->at(o);
  }
  OutputLabel const &at(MultiDiOutput const &o) const {
    return this->ptr->at(o);
  }
  OutputLabel const &at(MultiDiEdge const &e) const {
    return at(MultiDiOutput{e.src, e.srcIdx});
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const {
    return this->ptr->query_nodes(q);
  }
  std::unordered_set<MultiDiEdge> query_edges(MultiDiEdgeQuery const &q) const {
    return this->ptr->query_edges(q);
  }

  template <typename BaseImpl, typename N, typename O>
  static typename std::enable_if<std::conjunction<
                                  std::is_base_of<Interface, BaseImpl>,
                                  std::is_base_of<INodeLabel, N>,
                                  std::is_base_of<IOutputLabel, O>
                                >::value, OutputLabelledMultiDiGraph>::type
      create() {
    return OutputLabelledMultiDiGraph(
        make_cow_ptr<BaseImpl>(),
        make_cow_ptr<N>(),
        make_cow_ptr<O>());
  }

private:
  OutputLabelledMultiDiGraph(
      cow_ptr_t<Interface> ptr,
      cow_ptr_t<INodeLabel> nl,
      cow_ptr_t<IOutputLabel> ol)
      : ptr(ptr), nl(nl), ol(ol) {}

private:
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
