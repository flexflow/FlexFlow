#ifndef _FLEXFLOW_UTILS_GRAPH_LABELLED_NODE_LABELLED_OPEN
#define _FLEXFLOW_UTILS_GRAPH_LABELLED_NODE_LABELLED_OPEN

#include "utils/graph/open_graphs.h"

namespace FlexFlow {

template <typename NodeLabel>
struct INodeLabelledOpenMultiDiGraphView
    : virtual INodeLabelledMultiDiGraphView<NodeLabel>,
      virtual IOpenMultiDiGraphView {
  INodeLabelledOpenMultiDiGraphView() = default;
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
    return this->get_ptr().at(n);
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const {
    return this->get_ptr().query_nodes(q);
  }

  std::unordered_set<Edge> query_edges(OpenMultiDiEdgeQuery const &q) const {
    return this->get_ptr().query_edges(q);
  }

  template <typename BaseImpl, typename... Args>
  static typename std::enable_if<std::is_base_of<Interface, BaseImpl>::value,
                                 NodeLabelledOpenMultiDiGraphView>::type
      create(Args &&...args) {
    return NodeLabelledOpenMultiDiGraphView(
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

template <typename NodeLabel>
struct INodeLabelledOpenMultiDiGraph
    : virtual INodeLabelledOpenMultiDiGraphView<NodeLabel> {
  virtual Node add_node(NodeLabel const &) = 0;
  virtual NodePort add_node_port() = 0;
  virtual NodeLabel &at(Node const &) = 0;
  virtual void add_edge(OpenMultiDiEdge const &e) = 0;

  using INodeLabelledOpenMultiDiGraphView<NodeLabel>::at;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(INodeLabelledOpenMultiDiGraphView<int>);

template <typename NodeLabel>
struct NodeLabelledOpenMultiDiGraph
    : virtual NodeLabelledOpenMultiDiGraphView<NodeLabel> {
private:
  using Interface = INodeLabelledOpenMultiDiGraph<NodeLabel>;

public:
  NodeLabelledOpenMultiDiGraph(NodeLabelledOpenMultiDiGraph const &) = default;
  NodeLabelledOpenMultiDiGraph &
      operator=(NodeLabelledOpenMultiDiGraph const &) = default;

  NodeLabel &at(Node const &n) {
    return this->get_ptr().at(n);
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const {
    return this->get_ptr().query_nodes(q);
  }

  std::unordered_set<OpenMultiDiEdge>
      query_edges(OpenMultiDiEdge const &q) const {
    return this->get_ptr().query_edges(q);
  }

  Node add_node(NodeLabel const &l) {
    return this->get_ptr().add_node(l);
  }

  NodePort add_node_port() {
    return this->get_ptr().add_node_port();
  }

  void add_edge(OpenMultiDiEdge const &e) {
    return this->get_ptr().add_edge(e);
  }

  using NodeLabelledOpenMultiDiGraphView<NodeLabel>::at;

  template <typename BaseImpl>
  static typename std::enable_if<std::is_base_of<Interface, BaseImpl>::value,
                                 NodeLabelledOpenMultiDiGraph>::type
      create() {
    return NodeLabelledOpenMultiDiGraph(make_cow_ptr<BaseImpl>());
  }

private:
  NodeLabelledOpenMultiDiGraph(cow_ptr_t<Interface> ptr) : GraphView(ptr) {}

  Interface &get_ptr() {
    return *std::reinterpret_pointer_cast<Interface>(
        GraphView::ptr.get_mutable());
  }

  Interface const &get_ptr() const {
    return *std::reinterpret_pointer_cast<Interface const>(
        GraphView::ptr.get());
  }
};

} // namespace FlexFlow

#endif
