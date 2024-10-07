#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_FIND_PATHS_TO_LEAF_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_FIND_PATHS_TO_LEAF_H

#include "utils/full_binary_tree/find_paths_to_leaf.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree_implementation.h"

namespace FlexFlow {

template <typename Tree, typename Series, typename Parallel, typename Leaf>
std::unordered_set<BinaryTreePath>
    find_paths_to_leaf(Tree const &tree,
                       GenericBinarySPDecompositionTreeImplementation<Tree, Series, Parallel, Leaf> const &impl,
                       Leaf const &needle) {
  FullBinaryTreeImplementation<Tree, std::variant<Series, Parallel>, Leaf> 
    full_binary_impl = get_full_binary_impl_from_generic_sp_impl(impl);

  return find_paths_to_leaf(tree, full_binary_impl, needle);
}

} // namespace FlexFlow

#endif
