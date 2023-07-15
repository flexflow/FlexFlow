#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_LABELLED_OPEN_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_LABELLED_OPEN_H

#include "labelled_open_interfaces.h"
#include "node_labelled.h"
#include "utils/graph/open_graphs.h"

namespace FlexFlow {

template <typename NodeLabel,
          typename EdgeLabel,
          typename InputLabel = EdgeLabel,
          typename OutputLabel = InputLabel>
struct LabelledOpenMultiDiGraphView {
private:
  using Interface = ILabelledOpenMultiDiGraphView<NodeLabel,
                                                  EdgeLabel,
                                                  InputLabel,
                                                  OutputLabel>;

public:
  LabelledOpenMultiDiGraphView() = delete;

  operator OpenMultiDiGraphView() const;

  ILabelledOpenMultiDiGraphView<NodeLabel,
                                EdgeLabel,
                                InputLabel,
                                OutputLabel> const *
      unsafe() const {
    return this->ptr.get();
  }

private:
  std::shared_ptr<Interface const> ptr;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(
    LabelledOpenMultiDiGraphView<int, int, int, int>);

template <typename NodeLabel,
          typename EdgeLabel,
          typename InputLabel = EdgeLabel,
          typename OutputLabel = InputLabel>
struct LabelledOpenMultiDiGraph {
private:
  using Interface =
      ILabelledOpenMultiDiGraph<NodeLabel, EdgeLabel, InputLabel, OutputLabel>;

public:
  LabelledOpenMultiDiGraph() = delete;
  LabelledOpenMultiDiGraph(LabelledOpenMultiDiGraph const &other) = default;
  LabelledOpenMultiDiGraph &
      operator=(LabelledOpenMultiDiGraph const &other) = default;

  operator LabelledOpenMultiDiGraphView<NodeLabel,
                                        EdgeLabel,
                                        InputLabel,
                                        OutputLabel>() const;
  operator OpenMultiDiGraphView() const;

  friend void swap(LabelledOpenMultiDiGraph &lhs,
                   LabelledOpenMultiDiGraph &rhs) {
    using std::swap;

    swap(lhs.ptr, rhs.ptr);
  }

  Node add_node(NodeLabel const &l) {
    return this->ptr->add_node(l);
  }
  NodeLabel &at(Node const &n) {
    return this->ptr->at(n);
  }
  // NodeLabel const &at(Node const &n) const { return this->ptr->at(n); }
  NodeLabel const &at(Node const &n) const {
    return this->ptr->at(n);
  }

  void add_node_unsafe(Node const &n, NodeLabel const &l) {
    this->ptr->add_node_unsafe(n, l);
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const {
    return this->ptr->query_nodes(q);
  }
  std::unordered_set<OpenMultiDiEdge>
      query_edges(OpenMultiDiEdgeQuery const &q) const {
    return this->ptr->query_edges(q);
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

  void add_edge(InputMultiDiEdge const &e, InputLabel const &l) {
    return this->ptr->add_edge(e, l);
  }
  InputLabel &at(InputMultiDiEdge const &e) {
    return this->ptr->at(e);
  }
  InputLabel const &at(InputMultiDiEdge const &e) const {
    return this->ptr->at(e);
  }

  void add_edge(OutputMultiDiEdge const &, OutputLabel const &);
  OutputLabel &at(OutputMultiDiEdge const &);
  OutputLabel const &at(OutputMultiDiEdge const &) const;

  template <typename BaseImpl>
  static typename std::enable_if<std::is_base_of<Interface, BaseImpl>::value,
                                 LabelledOpenMultiDiGraph>::type
      create() {
    return LabelledOpenMultiDiGraph(make_unique<BaseImpl>());
  }

private:
  LabelledOpenMultiDiGraph(std::unique_ptr<Interface> ptr)
      : ptr(std::move(ptr)) {}

private:
  cow_ptr_t<Interface> ptr;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(
    LabelledOpenMultiDiGraph<int, int, int, int>);

} // namespace FlexFlow

#endif
