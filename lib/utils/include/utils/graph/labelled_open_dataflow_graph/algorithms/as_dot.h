#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_ALGORITHMS_AS_DOT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_ALGORITHMS_AS_DOT_H

#include "utils/graph/labelled_open_dataflow_graph/labelled_open_dataflow_graph_view.h"
#include "utils/graph/open_dataflow_graph/algorithms/as_dot.h"

namespace FlexFlow {

template <typename NodeLabel, typename ValueLabel>
std::string as_dot(
    LabelledOpenDataflowGraphView<NodeLabel, ValueLabel> const &g,
    std::function<std::string(NodeLabel const &)> const &get_node_label,
    std::function<std::string(ValueLabel const &)> const &get_input_label) {
  std::function<std::string(Node const &)> unlabelled_get_node_label =
      [&](Node const &n) -> std::string { return get_node_label(g.at(n)); };

  std::function<std::string(DataflowGraphInput const &)>
      unlabelled_get_input_label = [&](DataflowGraphInput const &i) {
        return get_input_label(g.at(OpenDataflowValue{i}));
      };

  return as_dot(static_cast<OpenDataflowGraphView>(g),
                unlabelled_get_node_label,
                unlabelled_get_input_label);
}

} // namespace FlexFlow

#endif
