#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_LABELLED_OPEN_INTERFACES_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_LABELLED_OPEN_INTERFACES_H

#include "standard_labelled_interfaces.h"
#include "utils/containers.h"
#include "utils/graph/open_graph_interfaces.h"

namespace FlexFlow {

template <typename NodeLabel,
          typename EdgeLabel,
          typename InputLabel = EdgeLabel,
          typename OutputLabel = InputLabel>
struct ILabelledOpenMultiDiGraphView
    : public IOpenMultiDiGraphView,
      public ILabelledMultiDiGraphView<NodeLabel, EdgeLabel> {
public:
  using IOpenMultiDiGraphView::query_edges; // Add this line

  std::unordered_set<MultiDiEdge> query_edges(MultiDiEdgeQuery const &q) const {
    // return IOpenMultiDiGraphView::query_edges(q);
    return IOpenMultiDiGraphView::query_edges(q);
  }

  using ILabelledMultiDiGraphView<NodeLabel, EdgeLabel>::at;
  virtual InputLabel const &at(InputMultiDiEdge const &e) const = 0;
  virtual OutputLabel const &at(OutputMultiDiEdge const &e) const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(
    ILabelledOpenMultiDiGraphView<int, int, int, int>);

template <typename NodeLabel,
          typename EdgeLabel,
          typename InputLabel = EdgeLabel,
          typename OutputLabel = InputLabel>
struct ILabelledOpenMultiDiGraph
    : public ILabelledMultiDiGraph<NodeLabel, EdgeLabel>,
      public ILabelledOpenMultiDiGraphView<NodeLabel,
                                           EdgeLabel,
                                           InputLabel,
                                           OutputLabel> {
public:
  virtual ILabelledOpenMultiDiGraph *clone() const = 0;

  virtual void add_edge(InputMultiDiEdge const &e, InputLabel const &label) = 0;
  virtual void add_edge(OutputMultiDiEdge const &e,
                        OutputLabel const &label) = 0;

  virtual InputLabel const &at(InputMultiDiEdge const &e) const = 0;
  virtual InputLabel &at(InputMultiDiEdge const &e) = 0;

  virtual OutputLabel const &at(OutputMultiDiEdge const &e) const = 0;
  virtual OutputLabel &at(OutputMultiDiEdge const &e) = 0;

  using ILabelledMultiDiGraph<NodeLabel, EdgeLabel>::add_node;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(ILabelledOpenMultiDiGraph<int, int, int, int>);

} // namespace FlexFlow

#endif
