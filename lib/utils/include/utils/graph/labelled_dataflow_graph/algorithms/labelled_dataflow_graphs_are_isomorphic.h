#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_DATAFLOW_GRAPH_ALGORITHMS_LABELLED_DATAFLOW_GRAPHS_ARE_ISOMORPHIC_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_DATAFLOW_GRAPH_ALGORITHMS_LABELLED_DATAFLOW_GRAPHS_ARE_ISOMORPHIC_H

#include "utils/graph/labelled_dataflow_graph/labelled_dataflow_graph.h"
#include "utils/graph/labelled_dataflow_graph/algorithms/find_isomorphism.h"

namespace FlexFlow {

template <typename NodeLabel, typename ValueLabel>
bool labelled_dataflow_graphs_are_isomorphic(LabelledDataflowGraph<NodeLabel, ValueLabel> const &src,
                                             LabelledDataflowGraph<NodeLabel, ValueLabel> const &dst) {
  return find_isomorphism(src, dst).has_value();
}


} // namespace FlexFlow

#endif
