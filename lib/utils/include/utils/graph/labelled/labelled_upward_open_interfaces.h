#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_LABELLED_UPWARD_OPEN_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_LABELLED_UPWARD_OPEN_H

#include "standard_labelled_interfaces.h"
#include "utils/graph/open_graph_interfaces.h"

namespace FlexFlow {

template <typename NodeLabel, typename EdgeLabel, typename InputLabel = EdgeLabel>
struct ILabelledUpwardOpenMultiDiGraphView : public ILabelledMultiDiGraphView<NodeLabel, EdgeLabel>,
                                             public IDownwardOpenMultiDiGraphView {
  virtual ~ILabelledUpwardOpenMultiDiGraphView() = default;

  using ILabelledMultiDiGraphView<NodeLabel, EdgeLabel>::at;
  virtual InputLabel const &at(InputMultiDiEdge const &) const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(ILabelledUpwardOpenMultiDiGraphView<int, int, int>);

template <typename NodeLabel, typename EdgeLabel, typename InputLabel = EdgeLabel>
struct ILabelledUpwardOpenMultiDiGraph : public ILabelledUpwardOpenMultiDiGraphView<NodeLabel, EdgeLabel, InputLabel>,
                                         public ILabelledMultiDiGraph<NodeLabel, EdgeLabel> {

  using ILabelledUpwardOpenMultiDiGraphView<NodeLabel, EdgeLabel, InputLabel>::at;
  using ILabelledMultiDiGraph<NodeLabel, EdgeLabel>::at;
  virtual InputLabel &at(InputLabel const &) = 0;

  using ILabelledMultiDiGraph<NodeLabel, EdgeLabel>::add_edge;
  virtual void add_edge(InputMultiDiEdge const &, InputLabel  const &) = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(ILabelledUpwardOpenMultiDiGraph<int, int, int>);

}

#endif
