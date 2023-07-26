#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_LABELLED_UPWARD_OPEN_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_LABELLED_UPWARD_OPEN_H

#include "labelled_open.h"
#include "labelled_upward_open_interfaces.h"
#include "utils/test_types.h"

namespace FlexFlow {

template <typename NodeLabel, typename EdgeLabel, typename InputLabel>
struct LabelledUpwardOpenMultiDiGraphView {
private:
  using Interface =
      ILabelledUpwardOpenMultiDiGraphView<NodeLabel, EdgeLabel, InputLabel>;

public:
  LabelledUpwardOpenMultiDiGraphView() = delete;

  template <typename OutputLabel>
  operator LabelledOpenMultiDiGraphView<NodeLabel,
                                        EdgeLabel,
                                        InputLabel,
                                        OutputLabel>() const {
    NOT_IMPLEMENTED();
  }

  InputLabel const &at(InputMultiDiEdge const &e) const {
    return this->ptr->at(e);
  }

  EdgeLabel const &at(MultiDiEdge const &e) const {
    return this->ptr->at(e);
  }

  NodeLabel const &at(Node const &n) const {
    return this->ptr->at(n);
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const {
    return this->ptr->query_nodes(q);
  }

  std::unordered_set<UpwardOpenMultiDiEdge>
      query_edges(UpwardOpenMultiDiEdgeQuery const &q) const {
    return this->ptr->query_edges(q);
  }

  template <typename BaseImpl, typename... Args>
  static typename std::enable_if<std::is_base_of<Interface, BaseImpl>::value,
                                 LabelledUpwardOpenMultiDiGraphView>::type
      create(Args &&...args) {
    return LabelledUpwardOpenMultiDiGraphView(
        std::make_shared<BaseImpl const>(std::forward<Args>(args)...));
  }

private:
  LabelledUpwardOpenMultiDiGraphView(
      std::shared_ptr<Interface const> const &ptr)
      : ptr(ptr) {}

  std::shared_ptr<Interface const> ptr;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(
    LabelledUpwardOpenMultiDiGraphView<test_types::cmp,
                                       test_types::cmp,
                                       test_types::cmp>);
CHECK_NOT_ABSTRACT(LabelledUpwardOpenMultiDiGraphView<test_types::cmp,
                                                      test_types::cmp,
                                                      test_types::cmp>);

template <typename NodeLabel, typename EdgeLabel, typename InputLabel>
struct LabelledUpwardOpenMultiDiGraph {
private:
  using Interface =
      ILabelledUpwardOpenMultiDiGraph<NodeLabel, EdgeLabel, InputLabel>;

public:
  LabelledUpwardOpenMultiDiGraph() = delete;

  template <typename OutputLabel>
  operator LabelledOpenMultiDiGraphView<NodeLabel,
                                        EdgeLabel,
                                        InputLabel,
                                        OutputLabel>() const;

  InputLabel const &at(InputMultiDiEdge const &e) const {
    return this->ptr->at(e);
  }

  InputLabel &at(InputMultiDiEdge const &e) {
    return this->ptr.get_mutable()->at(e);
  }

  EdgeLabel const &at(MultiDiEdge const &e) const {
    return this->ptr->at(e);
  }

  EdgeLabel &at(MultiDiEdge const &e) {
    return this->ptr.get_mutable()->at(e);
  }

  NodeLabel const &at(Node const &n) const {
    return this->ptr->at(n);
  }

  NodeLabel &at(Node const &n) {
    return this->ptr.get_mutable()->at(n);
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const {
    return this->ptr->query_nodes(q);
  }

  std::unordered_set<UpwardOpenMultiDiEdge>
      query_edges(UpwardOpenMultiDiEdgeQuery const &q) const {
    return this->ptr->query_edges(q);
  }

  Node add_node(NodeLabel const &l) {
    return this->ptr.get_mutable()->add_node(l);
  }

  void add_node_unsafe(Node const &n, NodeLabel const &l) {
    return this->ptr.get_mutable()->add_node_unsafe();
  }

  void add_edge(MultiDiEdge const &e, EdgeLabel const &l) {
    return this->ptr.get_mutable()->add_edge(e, l);
  }

  void add_edge(InputMultiDiEdge const &e, InputLabel const &l) {
    return this->ptr.get_mutable()->add_edge(e, l);
  }

  template <typename BaseImpl>
  static typename std::enable_if<std::is_base_of<Interface, BaseImpl>::value,
                                 LabelledUpwardOpenMultiDiGraph>::type
      create() {
    return LabelledUpwardOpenMultiDiGraph{make_unique<BaseImpl>()};
  }

private:
  cow_ptr_t<Interface> ptr;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(
    LabelledUpwardOpenMultiDiGraph<int, int, int>);

} // namespace FlexFlow

#endif
