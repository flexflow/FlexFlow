#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GET_NUM_TREE_NODES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GET_NUM_TREE_NODES_H

#include "utils/full_binary_tree/get_num_tree_nodes.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree_implementation.h"

namespace FlexFlow {

template <typename Tree, typename Series, typename Parallel, typename Leaf>
int get_num_tree_nodes(
    Tree const &tree,
    GenericBinarySPDecompositionTreeImplementation<Tree,
                                                   Series,
                                                   Parallel,
                                                   Leaf> const &impl) {

  FullBinaryTreeImplementation<Tree, std::variant<Series, Parallel>, Leaf>
      full_binary_impl = get_full_binary_impl_from_generic_sp_impl(impl);

  return get_num_tree_nodes(tree, full_binary_impl);
}

} // namespace FlexFlow

#endif
