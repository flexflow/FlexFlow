#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_leaves.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/is_binary_sp_tree_left_associative.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/is_binary_sp_tree_right_associative.h"

namespace FlexFlow {

GenericBinarySPDecompositionTreeImplementation<BinarySPDecompositionTree,
                                               BinarySeriesSplit,
                                               BinaryParallelSplit,
                                               Node>
    generic_impl_for_binary_sp_tree() {

  return GenericBinarySPDecompositionTreeImplementation<
      BinarySPDecompositionTree,
      BinarySeriesSplit,
      BinaryParallelSplit,
      Node>{
      /*series_get_left_child=*/[](BinarySeriesSplit const &split)
                                    -> BinarySPDecompositionTree const & {
        return split.get_left_child();
      },
      /*parallel_get_left_child=*/
      [](BinaryParallelSplit const &split)
          -> BinarySPDecompositionTree const & {
        return split.get_left_child();
      },
      /*series_get_right_child=*/
      [](BinarySeriesSplit const &split) -> BinarySPDecompositionTree const & {
        return split.get_right_child();
      },
      /*parallel_get_right_child=*/
      [](BinaryParallelSplit const &split)
          -> BinarySPDecompositionTree const & {
        return split.get_right_child();
      },
      /*get_node_type=*/
      [](BinarySPDecompositionTree const &tree) -> SPDecompositionTreeNodeType {
        return get_node_type(tree);
      },
      /*require_series=*/
      [](BinarySPDecompositionTree const &tree) -> BinarySeriesSplit const & {
        return tree.require_series();
      },
      /*require_parallel=*/
      [](BinarySPDecompositionTree const &tree) -> BinaryParallelSplit const & {
        return tree.require_parallel();
      },
      /*require_leaf=*/
      [](BinarySPDecompositionTree const &tree) -> Node const & {
        return tree.require_node();
      },
  };
}

bool is_binary_sp_tree_left_associative(BinarySPDecompositionTree const &tree) {
  return is_binary_sp_tree_left_associative(tree,
                                            generic_impl_for_binary_sp_tree());
}

bool is_binary_sp_tree_right_associative(
    BinarySPDecompositionTree const &tree) {
  return is_binary_sp_tree_right_associative(tree,
                                             generic_impl_for_binary_sp_tree());
}

std::unordered_multiset<Node>
    get_leaves(BinarySPDecompositionTree const &tree) {
  return get_leaves(tree, generic_impl_for_binary_sp_tree());
}

SPDecompositionTreeNodeType
    get_node_type(BinarySPDecompositionTree const &tree) {
  return tree.visit<SPDecompositionTreeNodeType>(overload{
      [](BinarySeriesSplit const &) {
        return SPDecompositionTreeNodeType::SERIES;
      },
      [](BinaryParallelSplit const &) {
        return SPDecompositionTreeNodeType::PARALLEL;
      },
      [](Node const &) { return SPDecompositionTreeNodeType::NODE; },
  });
}

} // namespace FlexFlow
