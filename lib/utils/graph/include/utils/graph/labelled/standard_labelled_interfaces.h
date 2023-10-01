#ifndef UTILS_GRAPH_INCLUDE_UTILS_GRAPH_LABELLED_STANDARD_LABELLED_INTERFACES
#define UTILS_GRAPH_INCLUDE_UTILS_GRAPH_LABELLED_STANDARD_LABELLED_INTERFACES

#include "node_labelled_interfaces.h"
#include "utils/graph/multidigraph.h"

namespace FlexFlow {

template <typename NodeLabel, typename EdgeLabel>
struct ILabelledMultiDiGraphView
    : public INodeLabelledMultiDiGraphView<NodeLabel> {
  ILabelledMultiDiGraphView() = default;
  ILabelledMultiDiGraphView(ILabelledMultiDiGraphView const &) = delete;
  ILabelledMultiDiGraphView &
      operator=(ILabelledMultiDiGraphView const &) = delete;

  virtual ~ILabelledMultiDiGraphView() = default;

  using INodeLabelledMultiDiGraphView<NodeLabel>::at;
  virtual EdgeLabel const &at(MultiDiEdge const &) const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(ILabelledMultiDiGraphView<int, int>);

template <typename NodeLabel, typename EdgeLabel>
struct ILabelledMultiDiGraph
    : public ILabelledMultiDiGraphView<NodeLabel, EdgeLabel> {
  ILabelledMultiDiGraph() = default;
  ILabelledMultiDiGraph(ILabelledMultiDiGraph const &) = delete;
  ILabelledMultiDiGraph &operator=(ILabelledMultiDiGraph const &) = delete;

  virtual ~ILabelledMultiDiGraph() = default;

  virtual ILabelledMultiDiGraph *clone() const = 0;

  using Edge = MultiDiEdge;
  using EdgeQuery = MultiDiEdgeQuery;

  using ILabelledMultiDiGraphView<NodeLabel, EdgeLabel>::at;
  virtual NodeLabel &at(Node const &) = 0;
  virtual EdgeLabel &at(MultiDiEdge const &) = 0;
  virtual void add_edge(MultiDiEdge const &, EdgeLabel const &) = 0;
  virtual Node add_node(NodeLabel const &) = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(ILabelledMultiDiGraph<int, int>);

} // namespace FlexFlow

#endif
