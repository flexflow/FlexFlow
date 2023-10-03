#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_ALGORITHMS_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_ALGORITHMS_H

#include "open_views.h"

namespace FlexFlow {

template <typename SubgraphView, typename NodeLabel, typename EdgeLabel>
OutputLabelledOpenMultiDiGraphView<NodeLabel, EdgeLabel> get_subgraph(OutputLabelledOpenMultiDiGraphView<NodeLabel, EdgeLabel> const &g, std::unordered_set<Node> const &nodes) {
  return OutputLabelledOpenMultiDiGraphView<NodeLabel, EdgeLabel>::create<OutputLabelledOpenMultiDiSubgraphView<SubgraphView, NodeLabel, EdgeLabel>>(g, nodes);
}

} // namespace FlexFlow

#endif
