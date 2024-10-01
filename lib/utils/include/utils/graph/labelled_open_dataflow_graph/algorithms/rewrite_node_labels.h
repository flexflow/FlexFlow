#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_ALGORITHMS_REWRITE_NODE_LABELS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_ALGORITHMS_REWRITE_NODE_LABELS_H

#include "utils/graph/labelled_open_dataflow_graph/algorithms/rewrite_labels.h"
#include "utils/overload.h"

namespace FlexFlow {

template <typename NodeLabel,
          typename ValueLabel,
          typename F,
          typename NewNodeLabel =
              std::invoke_result_t<F, Node const &, NodeLabel const &>>
LabelledOpenDataflowGraphView<NewNodeLabel, ValueLabel> rewrite_node_labels(
    LabelledOpenDataflowGraphView<NodeLabel, ValueLabel> const &g, F f) {
  return rewrite_labels(
      g,
      overload{
          [&](Node const &n, NodeLabel const &l) { return f(n, l); },
          [](OpenDataflowValue const &v, ValueLabel const &l) { return l; },
      });
}

} // namespace FlexFlow

#endif
