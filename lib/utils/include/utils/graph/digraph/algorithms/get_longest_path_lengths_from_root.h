#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_LONGEST_PATH_LENGTHS_FROM_ROOT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_LONGEST_PATH_LENGTHS_FROM_ROOT_H

#include "utils/graph/digraph/digraph_view.h"
#include <unordered_map>

namespace FlexFlow {

/**
 * @brief Computes the longest path lengths from the root in a single source,
 * directed acyclic graph.
 *
 * @return std::unordered_map<Node, int> For each node n, returns the length
 * (i.e. number of nodes) of the longest path from the root to n.
 *
 * @note The root has a path length of 1. g must be both acyclic and have a
 * single source.
 */
std::unordered_map<Node, int>
    get_longest_path_lengths_from_root(DiGraphView const &g);

/**
 * @brief Computes the weighted longest path lengths from the root in a single
 * source, directed acyclic graph.
 *
 * @return std::unordered_map<Node, float> For each node n, returns the length
 * (i.e. the sum of the weights of all the nodes) of the longest path from the
 * root to n.
 *
 * @note The root has a path length equal to its weight. g must be both acyclic
 * and have a single source.
 */
std::unordered_map<Node, float> get_weighted_longest_path_lengths_from_root(
    DiGraphView const &g, std::unordered_map<Node, float> const &node_costs);

} // namespace FlexFlow

#endif
