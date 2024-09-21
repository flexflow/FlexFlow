#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_BINARY_SP_DECOMPOSITION_TREE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_BINARY_SP_DECOMPOSITION_TREE_H

#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.dtg.h"
#include <unordered_set>

namespace FlexFlow {

BinarySPDecompositionTree make_series_split(BinarySPDecompositionTree const &,
                                            BinarySPDecompositionTree const &);
BinarySPDecompositionTree
    make_parallel_split(BinarySPDecompositionTree const &,
                        BinarySPDecompositionTree const &);
BinarySPDecompositionTree make_leaf_node(Node const &);

bool is_binary_sp_tree_left_associative(BinarySPDecompositionTree const &);
bool is_binary_sp_tree_right_associative(BinarySPDecompositionTree const &);

std::unordered_multiset<Node> get_leaves(BinarySPDecompositionTree const &);

} // namespace FlexFlow

#endif
