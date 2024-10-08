#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_GET_LEAVES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_GET_LEAVES_H

#include "utils/full_binary_tree/get_leaves.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree_implementation.h"

namespace FlexFlow {

template <typename Tree, typename Series, typename Parallel, typename Leaf>
std::unordered_multiset<Leaf> get_leaves(
    Tree const &tree,
    GenericBinarySPDecompositionTreeImplementation<Tree,
                                                   Series,
                                                   Parallel,
                                                   Leaf> const &impl) {

  FullBinaryTreeImplementation<Tree, std::variant<Series, Parallel>, Leaf>
      full_binary_impl = get_full_binary_impl_from_generic_sp_impl(impl);

  return get_leaves(tree, full_binary_impl);
}

} // namespace FlexFlow

#endif
