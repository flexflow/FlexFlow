#ifndef _FLEXFLOW_UTILS_GRAPH_LABELLED_NODE_LABELLED_OPEN
#define _FLEXFLOW_UTILS_GRAPH_LABELLED_NODE_LABELLED_OPEN

#include "node_labelled_open_interfaces.h"

namespace FlexFlow {

template <typename NodeLabel>
struct NodeLabelledOpenMultiDiGraphView : virtual MultiDiGraphView {
  using Interface = INodeLabelledOpenMultiDiGraphView<NodeLabel>;
};

template <typename NodeLabel>
struct NodeLabelledOpenMultiDiGraph : virtual NodeLabelledOpenMultiDiGraphView<NodeLabel> {
private:
  using Interface = IOpenMultiDiGraph<NodeLabel>;
  using INodeLabel = ILabel<Node, NodeLabel>;

public:
  NodeLabelledOpenMultiDiGraph() = delete;
  NodeLabelledOpenMultiDiGraph(NodeLabelledOpenMultiDiGraph const &) = default;
  NodeLabelledOpenMultiDiGraph &
      operator=(NodeLabelledOpenMultiDiGraph const &) = default;

  Node add_node(NodeLabel const &node);
  void add_edge(OpenMuldiDiEdge const &edge);

  NodeLabel &at(Node const &node);

  template <typename BaseImpl, typename N>
  static typename std::enable_if<std::conjunction<is_base_of<Interface, BaseImpl>,
                                                  is_base_of<INodeLabel, N>>::value,
                                 NodeLabelledOpenMultiDiGraph>::type
      create() {
    return NodeLabelledOpenMultiDiGraph(make_cow_ptr<BaseImpl>(),
                                make_cow_ptr<N>());
  }

private:
  NodeLabelledOpenMultiDiGraph(cow_ptr_t<Interface> ptr, cow_ptr_t<INodeLabel> nl)
    : NodeLabelledOpenMultiDiGraphView<NodeLabel>(ptr), nl(nl) {}

  cow_ptr_t<INodeLabel> nl;
};

} // namespace FlexFlow

#endif
