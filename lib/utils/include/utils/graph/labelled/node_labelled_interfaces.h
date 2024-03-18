#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_NODE_LABELLED_INTERFACES_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_NODE_LABELLED_INTERFACES_H

#include "utils/graph/multidigraph.h"

namespace FlexFlow {

template <typename NodeLabel>
struct INodeLabelledMultiDiGraphView : virtual public IMultiDiGraphView {
  INodeLabelledMultiDiGraphView() = default;
  INodeLabelledMultiDiGraphView(INodeLabelledMultiDiGraphView const &) = delete;
  INodeLabelledMultiDiGraphView &
      operator=(INodeLabelledMultiDiGraphView const &) = delete;

  virtual ~INodeLabelledMultiDiGraphView() {}

  virtual NodeLabel const &at(Node const &n) const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(INodeLabelledMultiDiGraphView<int>);

template <typename NodeLabel>
struct INodeLabelledMultiDiGraph
  : virtual INodeLabelledMultiDiGraphView<NodeLabel> {
  virtual NodeLabel &at(Node const &) = 0;
  virtual Node add_node(NodeLabel const &l) = 0;
  virtual NodePort add_node_port() = 0;
  virtual void add_edge(MultiDiEdge const &) = 0;

  virtual INodeLabelledMultiDiGraph *clone() const = 0;

  using INodeLabelledMultiDiGraphView<NodeLabel>::at;
};

} // namespace FlexFlow

#endif
