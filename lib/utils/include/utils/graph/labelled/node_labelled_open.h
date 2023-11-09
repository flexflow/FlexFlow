#ifndef _FLEXFLOW_UTILS_GRAPH_LABELLED_NODE_LABELLED_OPEN
#define _FLEXFLOW_UTILS_GRAPH_LABELLED_NODE_LABELLED_OPEN

#include "utils/graph/labelled/node_labelled.h"
#include "utils/graph/open_graphs.h"

namespace FlexFlow {

// todo: why INodeLabelledOpenMultiDiGraphView inherits from itself?
template <typename NodeLabel>
struct INodeLabelledOpenMultiDiGraphView
    : virtual INodeLabelledMultiDiGraphView<NodeLabel>,
      virtual IOpenMultiDiGraphView {
  INodeLabelledOpenMultiDiGraphView(INodeLabelledOpenMultiDiGraphView const &) =
      delete;
  INodeLabelledOpenMultiDiGraphView &
      operator=(INodeLabelledOpenMultiDiGraphView const &) = delete;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(INodeLabelledOpenMultiDiGraphView<int>);

template <typename NodeLabel>
struct NodeLabelledOpenMultiDiGraphView
    : virtual NodeLabelledMultiDiGraphView<NodeLabel>,
      virtual OpenMultiDiGraphView {
  using Interface = INodeLabelledOpenMultiDiGraphView<NodeLabel>;

public:
  // NodeLabelledOpenMultiDiGraphView() = delete;
  NodeLabelledOpenMultiDiGraphView(NodeLabelledOpenMultiDiGraphView const &) =
      default;
  NodeLabelledOpenMultiDiGraphView &
      operator=(NodeLabelledOpenMultiDiGraphView const &) = default;

  NodeLabel const &at(Node const &n) const {
    return this->get_ptr()->at(n);
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const {
    return this->get_ptr()->query_nodes(q);
  }

  std::unordered_set<Edge> query_edges(OpenMultiDiEdgeQuery const &q) const {
    return this->get_ptr()->query_edges(q);
  }

  template <typename BaseImpl, typename... Args>
  static typename std::enable_if<std::is_base_of<Interface, BaseImpl>::value,
                                 NodeLabelledOpenMultiDiGraphView>::type
      create(Args &&...args) {
    return NodeLabelledOpenMultiDiGraphView(
        make_cow_ptr<BaseImpl>(std::forward<Args>(args)...));
  }

protected:
  NodeLabelledOpenMultiDiGraphView()
      : NodeLabelledMultiDiGraphView<NodeLabel>(), OpenMultiDiGraphView() {}

private:
  cow_ptr_t<Interface> get_ptr() const {
    return cow_ptr_t(
        std::reinterpret_pointer_cast<Interface>(GraphView::ptr.get_mutable()));
  }
};

template <typename NodeLabel>
struct NodeLabelledOpenMultiDiGraph
    : virtual NodeLabelledOpenMultiDiGraphView<NodeLabel> {
private:
  using Interface = IOpenMultiDiGraph;
  using INodeLabel = ILabelling<Node, NodeLabel>;

public:
  // NodeLabelledOpenMultiDiGraph() = delete;
  NodeLabelledOpenMultiDiGraph(NodeLabelledOpenMultiDiGraph const &) = default;
  NodeLabelledOpenMultiDiGraph &
      operator=(NodeLabelledOpenMultiDiGraph const &) = default;

  NodeLabel const &at(Node const &n) const override {
    return nl->get_label(n);
  }

  NodeLabel &at(Node const &n) {
    return nl->get_label(n);
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const {
    return get_ptr()->query_nodes();
  }

  std::unordered_set<OpenMultiDiEdge>
      query_edges(OpenMultiDiEdgeQuery const &q) const {
    return get_ptr()->query_edges();
  }

  Node add_node(NodeLabel const &l) {
    Node n = MultiDiGraph::add_node();
    nl->add_label(n, l);
    return n;
  }

  NodePort add_node_port() {
    return get_ptr()->add_node_port();
  }

  void add_edge(OpenMultiDiEdge const &e) {
    return get_ptr()->add_edge(e);
  }

  template <typename BaseImpl, typename N>
  static typename std::enable_if<
      std::conjunction<std::is_base_of<Interface, BaseImpl>,
                       std::is_base_of<INodeLabel, N>>::value,
      NodeLabelledOpenMultiDiGraph>::type
      create() {
    return NodeLabelledOpenMultiDiGraph(make_cow_ptr<BaseImpl>(),
                                        make_cow_ptr<N>());
  }

private:
  NodeLabelledOpenMultiDiGraph(cow_ptr_t<Interface> ptr,
                               cow_ptr_t<INodeLabel> nl)
      : NodeLabelledOpenMultiDiGraphView<NodeLabel>(ptr), nl(nl) {}

  cow_ptr_t<Interface> get_ptr() const {
    return cow_ptr_t(
        std::reinterpret_pointer_cast<Interface>(GraphView::ptr.get_mutable()));
  }

  cow_ptr_t<INodeLabel> nl;
};

} // namespace FlexFlow

#endif
