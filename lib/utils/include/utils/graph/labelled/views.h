#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_VIEWS_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_VIEWS_H

#include "node_labelled_interfaces.h"
#include "standard_labelled_interfaces.h"

namespace FlexFlow {

template <typename NodeLabel>
struct NodeLabelledMultiDiSubgraphView
    : public INodeLabelledMultiDiGraphView<NodeLabel> {};

template <typename NodeLabel, typename EdgeLabel>
struct LabelledMultiDiSubgraphView
    : public ILabelledMultiDiGraphView<NodeLabel, EdgeLabel> {
public:
  LabelledMultiDiSubgraphView() = delete;
  template <typename InputLabel, typename OutputLabel>
  explicit LabelledMultiDiSubgraphView(
      ILabelledMultiDiGraphView<NodeLabel, EdgeLabel> const &,
      std::unordered_set<Node> const &);
};

} // namespace FlexFlow

#endif
