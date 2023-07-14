#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_NODE_LABELLED_INTERFACES_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_NODE_LABELLED_INTERFACES_H

#include "utils/graph/multidigraph.h"

namespace FlexFlow {

template <typename NodeLabel>
struct INodeLabelledMultiDiGraphView : public IMultiDiGraphView {
  INodeLabelledMultiDiGraphView(INodeLabelledMultiDiGraphView const &) = delete;
  INodeLabelledMultiDiGraphView &operator=(INodeLabelledMultiDiGraphView const &) = delete;

  virtual ~INodeLabelledMultiDiGraphView() {}

  virtual NodeLabel const &at(Node const &n) const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(INodeLabelledMultiDiGraphView<int>);

template <typename NodeLabel>
struct INodeLabelledMultiDiGraph : public INodeLabelledMultiDiGraphView<NodeLabel> {
public:
  INodeLabelledMultiDiGraph() = default;
  INodeLabelledMultiDiGraph(INodeLabelledMultiDiGraph const &) = delete;
  INodeLabelledMultiDiGraph &
      operator=(INodeLabelledMultiDiGraph const &) = delete;
  virtual ~INodeLabelledMultiDiGraph() {}

  virtual INodeLabelledMultiDiGraph *clone() const = 0;

  virtual void add_edge(MultiDiEdge const &) = 0;

  virtual Node add_node(NodeLabel const &) = 0;
  virtual NodeLabel &at(Node const &n) = 0;
  virtual NodeLabel const &at(Node const &n) const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(INodeLabelledMultiDiGraph<int>);

}

#endif
