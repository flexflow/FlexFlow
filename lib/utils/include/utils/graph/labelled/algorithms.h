#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_ALGORITHMS_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_ALGORITHMS_H

#include "labelled_downward_open.h"
#include "labelled_open.h"
#include "labelled_upward_open.h"
#include "node_labelled.h"
#include "output_labelled.h"
#include "standard_labelled.h"
#include "views.h"

namespace FlexFlow {

template <typename NodeLabel>
NodeLabelledMultiDiGraphView<NodeLabel>
    get_subgraph(NodeLabelledMultiDiGraphView<NodeLabel> const &g,
                 std::unordered_set<Node> const &nodes) {
  return NodeLabelledMultiDiGraphView<NodeLabel>::template create<
      NodeLabelledMultiDiSubgraphView<NodeLabel>>(nodes);
}

template <typename NodeLabel, typename EdgeLabel>
LabelledMultiDiGraphView<NodeLabel, EdgeLabel>
    get_subgraph(LabelledMultiDiGraphView<NodeLabel, EdgeLabel> const &g,
                 std::unordered_set<Node> const &nodes) {
  return LabelledMultiDiGraphView<NodeLabel, EdgeLabel>::template create<
      LabelledMultiDiSubgraphView<NodeLabel, EdgeLabel>>(nodes);
}

template <typename NodeLabel, typename OutputLabel>
OutputLabelledMultiDiGraphView<NodeLabel, OutputLabel> get_subgraph(
    OutputLabelledMultiDiGraphView<NodeLabel, OutputLabel> const &g,
    std::unordered_set<Node> const &nodes) {
  return OutputLabelledMultiDiGraphView<NodeLabel, OutputLabel>::
      template create<OutputLabelledMultiDiGraphView<NodeLabel, OutputLabel>>(
          g, nodes);
}

} // namespace FlexFlow

#endif
