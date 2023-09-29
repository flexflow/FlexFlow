#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_LABELLED_DONWARD_OPEN_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_LABELLED_DONWARD_OPEN_H

#include "labelled_downward_open_interfaces.h"
#include "labelled_open.h"

namespace FlexFlow {

template <typename NodeLabel, typename EdgeLabel, typename OutputLabel>
struct LabelledDownwardOpenMultiDiGraphView : virtual LabelledMultiDiGraphView<NodeLabel, EdgeLabel>,
                                              virtual DownwardOpenMultiDiGraphView {
private:
  using Interface =
      ILabelledDownwardOpenMultiDiGraphView<NodeLabel, EdgeLabel, OutputLabel>;

public:
  template <typename InputLabel>
  operator LabelledOpenMultiDiGraphView<NodeLabel,
                                        EdgeLabel,
                                        InputLabel,
                                        OutputLabel>() const;

  OutputLabel const &at(OutputMultiDiEdge const &e) const {
    return this->get_ptr()->at(e);
  }

  EdgeLabel const &at(MultiDiEdge const &e) const {
    return this->get_ptr()->at(e);
  }

  NodeLabel const &at(Node const &n) const {
    return this->get_ptr()->at(n);
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const {
    return this->get_ptr()->query_nodes(q);
  }

  std::unordered_set<DownwardOpenMultiDiEdge>
      query_edges(DownwardOpenMultiDiEdgeQuery const &q) const {
    return this->get_ptr()->query_edges(q);
  }

  template <typename BaseImpl, typename... Args>
  static typename std::enable_if<std::is_base_of<Interface, BaseImpl>::value,
                                 LabelledDownwardOpenMultiDiGraphView>::type
      create(Args &&...args) {
    return LabelledDownwardOpenMultiDiGraphView(
        std::make_shared<BaseImpl const>(std::forward<Args>(args)...));
  }

private:
  LabelledDownardOpenMultiDiGraphView(std::shared_ptr<Interface const> ptr) : DownwardOpenMultiDiGraphView(ptr) {}
  std::shared_ptr<Interface const> get_ptr() const {
    return static_cast<std::shared_ptr<Interface const>>(ptr);
  }
};

template <typename NodeLabel, typename EdgeLabel, typename OutputLabel>
struct LabelledDownardOpenMultiDiGraph {
private:
  using Interface =
      ILabelledDownwardOpenMultiDiGraph<NodeLabel, EdgeLabel, OutputLabel>;

public:
  OutputLabel const &at(OutputMultiDiEdge const &e) const {
    return this->ptr->at(e);
  }

  OutputLabel &at(OutputMultiDiEdge const &e) {
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

  std::unordered_set<DownwardOpenMultiDiEdge>
      query_edges(DownwardOpenMultiDiEdgeQuery const &q) const {
    return this->ptr->query_edges(q);
  }

  Node add_node(NodeLabel const &l) {
    return this->ptr.get_mutable()->add_node(l);
  }

  void add_node_unsafe(Node const &n) {
    return this->ptr.get_mutable()->add_node_unsafe(n);
  }

  void remove_node_unsafe(Node const &n) {
    return this->ptr.get_mutable()->remove_node_unsafe(n);
  }

  void add_edge(MultiDiEdge const &e, EdgeLabel const &l) {
    return this->ptr.get_mutable()->add_edge(e, l);
  }

  void add_edge(OutputMultiDiEdge const &e, OutputLabel const &l) {
    return this->ptr.get_mutable()->add_edge(e, l);
  }

  void remove_edge(OutputMultiDiEdge const &e) {
    return this->ptr.get_mutable()->remove_edge(e);
  }

  void remove_edge(MultiDiEdge const &e) {
    return this->ptr.get_mutable()->remove_edge(e);
  }

  template <typename BaseImpl>
  static typename std::enable_if<std::is_base_of<Interface, BaseImpl>::value,
                                 LabelledDownardOpenMultiDiGraph>::type
      create() {
    return LabelledDownardOpenMultiDiGraph(make_unique<BaseImpl>());
  }

private:
  cow_ptr_t<Interface> ptr;
};

} // namespace FlexFlow

#endif
