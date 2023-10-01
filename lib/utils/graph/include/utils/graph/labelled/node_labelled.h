#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_NODE_LABELLED_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_NODE_LABELLED_H

#include "node_labelled_interfaces.h"
#include "label.h"

namespace FlexFlow {

template <typename NodeLabel>
struct NodeLabelledMultiDiGraphView : virtual public MultiDiGraphView {
private:
  using Interface = INodeLabelledMultiDiGraphView<int>;

public:
  NodeLabelledMultiDiGraphView() = delete;
  NodeLabelledMultiDiGraphView(NodeLabelledMultiDiGraphView const &) = default;
  NodeLabelledMultiDiGraphView &
      operator=(NodeLabelledMultiDiGraphView const &) = default;

  NodeLabel const &at(Node const &n) const {
    return get_ptr()->at(n);
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const {
    return get_ptr()->query_nodes(q);
  }

  std::unordered_set<MultiDiEdge> query_edges(MultiDiEdgeQuery const &q) const {
    return get_ptr()->query_edges(q);
  }

  template <typename BaseImpl, typename... Args>
  static typename std::enable_if<std::is_base_of<Interface, BaseImpl>::value,
                                 NodeLabelledMultiDiGraphView>::type
      create(Args &&...args) {
    return NodeLabelledMultiDiGraphView(
        std::make_shared<BaseImpl>(std::forward<Args>(args)...));
  }

protected:
  NodeLabelledMultiDiGraphView(std::shared_ptr<Interface const> ptr) : MultiDiGraphView(ptr) {}
  std::share_ptr<Interface const> get_ptr() const {
    return static_cast<std::shared_ptr<Interface const>>(ptr);
  }
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(NodeLabelledMultiDiGraphView<int>);

template <typename NodeLabel>
struct NodeLabelledMultiDiGraph : virtual NodeLabelledMultiDiGraphView<NodeLabel> {
private:
  using GraphIf = IMultiDiGraph<NodeLabel>;
  using NodeLabelIf = ILabel<Node, NodeLabel>;

public:
  NodeLabelledMultiDiGraph() = delete;
  NodeLabelledMultiDiGraph(NodeLabelledMultiDiGraph const &) = default;
  NodeLabelledMultiDiGraph &
      operator=(NodeLabelledMultiDiGraph const &) = default;

  friend void swap(NodeLabelledMultiDiGraph &lhs,
                   NodeLabelledMultiDiGraph &rhs) {
    using std::swap;

    swap(lhs.ptr, rhs.ptr);
  }

  Node add_node(NodeLabel const &l) {
    Node n = MultiDiGraph::add_node();
    nl->add_label(n, l);
    return n;
  }
  NodeLabel &at(Node const &n) {
    return nl->get_label(n);
  }

  void add_edge(MultiDiEdge const &e) {
    return get_ptr()->add_edge(e);
  }

  template <typename GraphImpl, typename NLImpl>
  static typename std::enable_if<std::conjunction<std::is_base_of<GraphIf, GraphImpl>,
                                                  std::is_base_of<NodeLabelIf, NLImpl>>::value,
                                 NodeLabelledMultiDiGraph>::type
      create() {
    return NodeLabelledMultiDiGraph(make_cow_ptr<GraphImpl>(), make_cow_ptr<NLImpl>());
  }

protected:
  NodeLabelledMultiDiGraph(cow_ptr_t<GraphIf> ptr, cow_ptr_t<NodeLabelIf> nl)
      : NodeLabelledMultiDiGraphView<NodeLabel>(ptr), nl(nl) {}

  cow_ptr_t<NodeLabelIf> nl;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(NodeLabelledMultiDiGraph<int>);

} // namespace FlexFlow

#endif
