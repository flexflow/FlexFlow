#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_ALGORITHMS_LABELLED_OPEN_DATAFLOW_GRAPHS_ARE_ISOMORPHIC_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_ALGORITHMS_LABELLED_OPEN_DATAFLOW_GRAPHS_ARE_ISOMORPHIC_H

#include "utils/graph/labelled_open_dataflow_graph/labelled_open_dataflow_graph_view.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/find_isomorphism.h"

namespace FlexFlow {

template <typename NodeLabel, typename ValueLabel>
bool labelled_open_dataflow_graphs_are_isomorphic(LabelledOpenDataflowGraphView<NodeLabel, ValueLabel> const &lhs,
                                                  LabelledOpenDataflowGraphView<NodeLabel, ValueLabel> const &rhs) {
  return find_isomorphism(lhs, rhs).has_value();
}

} // namespace FlexFlow

#endif
