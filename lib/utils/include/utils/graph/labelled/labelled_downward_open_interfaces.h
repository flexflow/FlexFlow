#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_LABELLED_DOWNWARD_OPEN_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_LABELLED_DOWNWARD_OPEN_H

#include "standard_labelled_interfaces.h"
#include "utils/graph/open_graph_interfaces.h"

namespace FlexFlow {

template <typename NodeLabel,
          typename EdgeLabel,
          typename OutputLabel = EdgeLabel>
struct ILabelledDownwardOpenMultiDiGraphView
    : public ILabelledMultiDiGraphView<NodeLabel, EdgeLabel>,
      public IDownwardOpenMultiDiGraphView {
  virtual ~ILabelledDownwardOpenMultiDiGraphView() = default;

  using ILabelledMultiDiGraphView<NodeLabel, EdgeLabel>::at;
  virtual OutputLabel const &at(OutputMultiDiEdge const &) const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(
    ILabelledDownwardOpenMultiDiGraphView<int, int, int>);

template <typename NodeLabel,
          typename EdgeLabel,
          typename OutputLabel = EdgeLabel>
struct ILabelledDownwardOpenMultiDiGraph
    : public ILabelledDownwardOpenMultiDiGraphView<NodeLabel,
                                                   EdgeLabel,
                                                   OutputLabel>,
      public ILabelledMultiDiGraph<NodeLabel, EdgeLabel> {

  using ILabelledDownwardOpenMultiDiGraphView<NodeLabel,
                                              EdgeLabel,
                                              OutputLabel>::at;
  using ILabelledMultiDiGraph<NodeLabel, EdgeLabel>::at;
  virtual OutputLabel &at(OutputLabel const &) = 0;

  using ILabelledMultiDiGraph<NodeLabel, EdgeLabel>::add_edge;
  virtual void add_edge(OutputMultiDiEdge const &, OutputLabel const &) = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(
    ILabelledDownwardOpenMultiDiGraph<int, int, int>);

} // namespace FlexFlow

#endif
