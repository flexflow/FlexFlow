#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_LONGEST_PATH_LENGHTS_FROM_ROOT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_LONGEST_PATH_LENGHTS_FROM_ROOT_H

#include "utils/graph/digraph/digraph_view.h"
#include <unordered_map>

namespace FlexFlow {

std::unordered_map<Node, float> get_weighted_longest_path_lengths_from_root(
    DiGraphView const &g, std::unordered_map<Node, float> const &node_costs);

std::unordered_map<Node, int>
    get_longest_path_lengths_from_root(DiGraphView const &g);

} // namespace FlexFlow

#endif
