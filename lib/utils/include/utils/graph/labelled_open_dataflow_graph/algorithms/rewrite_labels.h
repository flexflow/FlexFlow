#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_ALGORITHMS_REWRITE_LABELS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_ALGORITHMS_REWRITE_LABELS_H

#include "utils/graph/labelled_open_dataflow_graph/labelled_open_dataflow_graph_view.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/with_labelling.h"

namespace FlexFlow {

template <typename NodeLabel, 
          typename ValueLabel,
          typename F,
          typename NewNodeLabel = std::invoke_result_t<F, Node const &, NodeLabel const &>,
          typename NewValueLabel = std::invoke_result_t<F, OpenDataflowValue const &, ValueLabel const &>>
LabelledOpenDataflowGraphView<NewNodeLabel, NewValueLabel>
  rewrite_labels(LabelledOpenDataflowGraphView<NodeLabel, ValueLabel> const &g, F f) {
  std::unordered_map<Node, NewNodeLabel> node_labels = generate_map(get_nodes(g), f);
  std::unordered_map<OpenDataflowValue, NewValueLabel> value_labels = generate_map(get_nodes(g), f);
  return with_labelling(g, node_labels, value_labels);
}


} // namespace FlexFlow

#endif
