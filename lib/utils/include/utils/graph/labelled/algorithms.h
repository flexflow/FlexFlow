#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_ALGORITHMS_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_ALGORITHMS_H

#include "labelled_open.h"
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

enum class InputSettings { INCLUDE_INPUTS, EXCLUDE_INPUTS };
enum class OutputSettings { INCLUDE_OUTPUTS, EXCLUDE_OUTPUTS };

template <InputSettings INPUT_SETTINGS,
          OutputSettings OUTPUT_SETTINGS,
          typename NodeLabel,
          typename EdgeLabel,
          typename InputLabel,
          typename OutputLabel>
LabelledOpenMultiDiGraphView<NodeLabel, EdgeLabel, InputLabel, OutputLabel>
    get_subgraph(LabelledOpenMultiDiGraphView<NodeLabel,
                                              EdgeLabel,
                                              InputLabel,
                                              OutputLabel> const &g,
                 std::unordered_set<Node> const &nodes);

return LabelledOpenMultiDiGraphView<
    NodeLabel,
    EdgeLabel,
    InputLabel,
    OutputLabel>::template create<LabelledOpenMultiDiSubgraphView<NodeLabel,
                                                                  EdgeLabel,
                                                                  InputLabel,
                                                                  OutputLabel>>(
    g, nodes);
}
}

#endif
