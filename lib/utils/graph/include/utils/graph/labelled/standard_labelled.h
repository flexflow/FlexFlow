#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_STANDARD_LABELLED_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_STANDARD_LABELLED_H

#include "labelled_downward_open.h"
#include "labelled_open.h"
#include "labelled_upward_open.h"
#include "node_labelled.h"
#include "standard_labelled_interfaces.h"

namespace FlexFlow {

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
        std::make_shared<BaseImpl>(std::forward<Args>(args)...));
  }

protected:
  LabelledMultiDiGraphView(std::shared_ptr<Interface const> ptr) : NodeLabelledMultiDiGraphView<NodeLabel>(ptr) {}
  std::shared_ptr<Interface const> get_ptr() const {
    return static_assert<std::shared_ptr<Interface const>>(ptr);
  }
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(LabelledMultiDiGraphView<int, int>);

template <typename NodeLabel, typename EdgeLabel>
struct LabelledMultiDiGraph {
private:
  using Interface = ILabelledMultiDiGraph<NodeLabel, EdgeLabel>;

public:
  LabelledMultiDiGraph() = delete;
  LabelledMultiDiGraph(LabelledMultiDiGraph const &other) = default;
  LabelledMultiDiGraph &operator=(LabelledMultiDiGraph other) = default;

  friend void swap(LabelledMultiDiGraph &lhs, LabelledMultiDiGraph &rhs) {
    using std::swap;

    swap(lhs.ptr, rhs.ptr);
  }

  operator LabelledMultiDiGraphView() const;

  Node add_node(NodeLabel const &l) {
    return this->ptr->add_node(l);
  }

  NodeLabel &at(Node const &n) {
    return this->ptr.get_mutable()->at(n);
  }

  NodeLabel const &at(Node const &n) const {
    return this->ptr->at(n);
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

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const {
    return this->ptr->query_nodes(q);
  }
  std::unordered_set<MultiDiEdge> query_edges(MultiDiEdgeQuery const &q) const {
    return this->ptr->query_edges(q);
  }

  template <typename BaseImpl>
  static typename std::enable_if<std::is_base_of<Interface, BaseImpl>::value,
                                 LabelledMultiDiGraph>::type
      create() {
    return LabelledMultiDiGraph(make_unique<BaseImpl>());
  }

private:
  LabelledMultiDiGraph(cow_ptr_t<Interface> ptr) : ptr(ptr) {}

private:
  cow_ptr_t<Interface> ptr;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(LabelledMultiDiGraph<int, int>);

} // namespace FlexFlow

#endif
