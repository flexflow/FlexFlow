#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_ALGORITHMS_IS_ISOMORPHIC_UNDER_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_ALGORITHMS_IS_ISOMORPHIC_UNDER_H

#include "utils/graph/labelled_open_dataflow_graph/labelled_open_dataflow_graph_view.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/permute_node_ids.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/get_graph_data.h"

namespace FlexFlow {

template <typename NodeLabel, typename EdgeLabel>
bool is_isomorphic_under(LabelledOpenDataflowGraphView<NodeLabel, EdgeLabel> const &src,
                         LabelledOpenDataflowGraphView<NodeLabel, EdgeLabel> const &dst,
                         bidict<Node, Node> const &src_to_dst) {

  bidict<NewNode, Node> permutation = map_values(src_to_dst, [](Node const &dst_node) { return NewNode{dst_node}; }).reversed();
  return get_graph_data(permute_node_ids(src, permutation)) == get_graph_data(dst);
}

} // namespace FlexFlow

#endif
