#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OUTPUT_LABELLED_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OUTPUT_LABELLED_H

#include "output_labelled_interfaces.h"
#include "standard_labelled.h"

namespace FlexFlow {

template <typename NodeLabel, typename OutputLabel>
struct OutputLabelledMultiDiGraphView {
private:
  using Interface = IOutputLabelledMultiDiGraphView<NodeLabel, OutputLabel>;

public:
  OutputLabelledMultiDiGraphView() = delete;

  operator LabelledMultiDiGraphView<NodeLabel, OutputLabel>();

  NodeLabel const &at(Node const &n) const {
    return this->ptr->at(n);
  }

  OutputLabel const &at(MultiDiOutput const &o) const {
    return this->ptr->at(o);
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const {
    return this->ptr->query_nodes(q);
  }

  std::unordered_set<MultiDiEdge> query_edges(MultiDiEdgeQuery const &q) const {
    return this->ptr->query_edges(q);
  }

private:
  std::shared_ptr<Interface> ptr;
};

template <typename NodeLabel, typename OutputLabel>
struct OutputLabelledMultiDiGraph {
private:
  using Interface = IOutputLabelledMultiDiGraph<NodeLabel, OutputLabel>;

public:
  OutputLabelledMultiDiGraph() = delete;
  OutputLabelledMultiDiGraph(OutputLabelledMultiDiGraph const &other) = default;
  OutputLabelledMultiDiGraph &
      operator=(OutputLabelledMultiDiGraph const &other) = default;

  operator LabelledMultiDiGraphView<NodeLabel, OutputLabel>() const;
  operator MultiDiGraphView() const;
  operator GraphView() const;

  friend void swap(OutputLabelledMultiDiGraph &lhs,
                   OutputLabelledMultiDiGraph &rhs) {
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

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const {
    return this->ptr->query_nodes(q);
  }
  std::unordered_set<MultiDiEdge> query_edges(MultiDiEdgeQuery const &q) const {
    return this->ptr->query_edges(q);
  }

private:
  OutputLabelledMultiDiGraph(
      std::unique_ptr<IOutputLabelledMultiDiGraph<NodeLabel, OutputLabel>> ptr)
      : ptr(std::move(ptr)) {}

private:
  cow_ptr_t<Interface> ptr;
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
