#include "compiler/series_parallel/pcg/pcg_binary_sp_decomposition.h"
#include "compiler/series_parallel/pcg/pcg_binary_series_split.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_leaves.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/find_paths_to_leaf.h"
#include "utils/overload.h"

namespace FlexFlow {

GenericBinarySPDecompositionTreeImplementation<
  PCGBinarySPDecomposition,
  PCGBinarySeriesSplit,
  PCGBinaryParallelSplit,
  parallel_layer_guid_t> generic_impl_for_pcg_sp_tree() {

  return GenericBinarySPDecompositionTreeImplementation<
    PCGBinarySPDecomposition, 
    PCGBinarySeriesSplit,
    PCGBinaryParallelSplit,
    parallel_layer_guid_t>{
    /*series_get_left_child=*/[](PCGBinarySeriesSplit const &split) -> PCGBinarySPDecomposition const & {
      return split.get_left_child(); 
    },
    /*parallel_get_left_child=*/[](PCGBinaryParallelSplit const &split) -> PCGBinarySPDecomposition const & {
      return split.get_left_child();
    },
    /*series_get_right_child=*/[](PCGBinarySeriesSplit const &split) -> PCGBinarySPDecomposition const & {
      return split.get_right_child();
    },
    /*parallel_get_right_child=*/[](PCGBinaryParallelSplit const &split) -> PCGBinarySPDecomposition const & {
      return split.get_right_child();
    },
    /*get_node_type=*/[](PCGBinarySPDecomposition const &tree) -> SPDecompositionTreeNodeType {
      return get_node_type(tree);
    },
    /*require_series=*/[](PCGBinarySPDecomposition const &tree) -> PCGBinarySeriesSplit const & {
      return tree.get<PCGBinarySeriesSplit>();
    },
    /*require_parallel=*/[](PCGBinarySPDecomposition const &tree) -> PCGBinaryParallelSplit const & {
      return tree.get<PCGBinaryParallelSplit>();
    },
    /*require_leaf=*/[](PCGBinarySPDecomposition const &tree) -> parallel_layer_guid_t const & {
      return tree.get<parallel_layer_guid_t>();
    },
  };
}


BinarySPDecompositionTree binary_sp_tree_from_pcg_sp_tree(PCGBinarySPDecomposition const &pcg_tree) {
  return pcg_tree.visit<BinarySPDecompositionTree>(overload {
    [](PCGBinarySeriesSplit const &series) -> BinarySPDecompositionTree {
      return BinarySPDecompositionTree{
        binary_series_split_from_pcg_series_split(series),
      };
    },
    [](PCGBinaryParallelSplit const &parallel) -> BinarySPDecompositionTree {
      return BinarySPDecompositionTree{
        BinaryParallelSplit{
          binary_sp_tree_from_pcg_sp_tree(parallel.get_left_child()),
          binary_sp_tree_from_pcg_sp_tree(parallel.get_right_child()),
        },
      };
    },
    [](parallel_layer_guid_t const &layer) -> BinarySPDecompositionTree {
      return BinarySPDecompositionTree{
        layer.raw_graph_node,
      };
    },
  });
}

std::optional<PCGBinarySPDecomposition>
    get_pcg_balanced_binary_sp_decomposition(ParallelComputationGraph const &) {
  NOT_IMPLEMENTED();
}

std::unordered_multiset<parallel_layer_guid_t>
    get_parallel_layers(PCGBinarySPDecomposition const &tree) {
  return get_leaves(tree, generic_impl_for_pcg_sp_tree());
}

SPDecompositionTreeNodeType get_node_type(PCGBinarySPDecomposition const &tree) {
  return tree.visit<SPDecompositionTreeNodeType>(overload {
    [](PCGBinarySeriesSplit const &) { return SPDecompositionTreeNodeType::SERIES; },
    [](PCGBinaryParallelSplit const &) { return SPDecompositionTreeNodeType::PARALLEL; },
    [](parallel_layer_guid_t const &) { return SPDecompositionTreeNodeType::NODE; },
  });
}

std::unordered_set<BinaryTreePath>
    find_paths_to_leaf(PCGBinarySPDecomposition const &tree,
                       parallel_layer_guid_t const &leaf) {
  return find_paths_to_leaf(tree, generic_impl_for_pcg_sp_tree(), leaf);
}

} // namespace FlexFlow
