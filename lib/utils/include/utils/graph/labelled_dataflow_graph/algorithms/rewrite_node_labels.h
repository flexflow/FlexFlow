#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_DATAFLOW_GRAPH_ALGORITHMS_REWRITE_NODE_LABELS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_DATAFLOW_GRAPH_ALGORITHMS_REWRITE_NODE_LABELS_H

#include "utils/graph/labelled_open_dataflow_graph/algorithms/rewrite_node_labels.h"

namespace FlexFlow {

template <typename NodeLabel,
          typename ValueLabel,
          typename F,
          typename NewNodeLabel =
              std::invoke_result_t<F, Node const &, NodeLabel const &>>
LabelledDataflowGraphView<NewNodeLabel, ValueLabel> rewrite_node_labels(
    LabelledDataflowGraphView<NodeLabel, ValueLabel> const &g, F f) {
  return rewrite_node_labels<NodeLabel, ValueLabel, F, NewNodeLabel>(
    view_as_labelled_open_dataflow_graph(g), f);
}


} // namespace FlexFlow

#endif
