#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_ALGORITHMS_IS_ISOMORPHIC_UNDER_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_ALGORITHMS_IS_ISOMORPHIC_UNDER_H

#include "utils/graph/labelled_open_dataflow_graph/labelled_open_dataflow_graph_view.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/permute_node_ids.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/permute_input_ids.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/get_graph_data.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/open_dataflow_graph_isomorphism.dtg.h"
#include "utils/graph/open_dataflow_graph/algorithms/new_dataflow_graph_input.dtg.h"

namespace FlexFlow {

template <typename NodeLabel, typename EdgeLabel>
bool is_isomorphic_under(LabelledOpenDataflowGraphView<NodeLabel, EdgeLabel> const &src,
                         LabelledOpenDataflowGraphView<NodeLabel, EdgeLabel> const &dst,
                         OpenDataflowGraphIsomorphism const &candidate_isomorphism) {

  bidict<NewNode, Node> node_permutation = map_values(candidate_isomorphism.node_mapping, 
                                                      [](Node const &dst_node) { return NewNode{dst_node}; }
                                                      ).reversed();
  bidict<NewDataflowGraphInput, DataflowGraphInput> input_permutation = map_values(candidate_isomorphism.input_mapping,
                                                                                   [](DataflowGraphInput const &dst_input) { return NewDataflowGraphInput{dst_input}; }
                                                                                  ).reversed();
  return get_graph_data(permute_input_ids(permute_node_ids(src, node_permutation), input_permutation)) == get_graph_data(dst);
}

} // namespace FlexFlow

#endif
