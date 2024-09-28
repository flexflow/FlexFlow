#include "compiler/series_parallel/pcg_binary_sp_decomposition.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/make.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_node_type.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_leaves.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/require.h"

namespace FlexFlow {

std::optional<PCGBinarySPDecomposition>
  get_pcg_balanced_binary_sp_decomposition(ParallelComputationGraph const &) {
  NOT_IMPLEMENTED();
}

std::unordered_multiset<parallel_layer_guid_t>
  get_parallel_layers(PCGBinarySPDecomposition const &d) {
  return get_leaves(d.raw_tree);
}

SPDecompositionTreeNodeType get_node_type(PCGBinarySPDecomposition const &d) {
  return get_node_type(d.raw_tree);
}

PCGBinarySPDecomposition make_pcg_series_split(PCGBinarySPDecomposition const &lhs, PCGBinarySPDecomposition const &rhs) {
  return PCGBinarySPDecomposition{
      make_generic_binary_series_split(lhs.raw_tree, rhs.raw_tree),
  };
}

PCGBinarySPDecomposition make_pcg_parallel_split(PCGBinarySPDecomposition const &lhs, PCGBinarySPDecomposition const &rhs) {
  return PCGBinarySPDecomposition{
      make_generic_binary_parallel_split(lhs.raw_tree, rhs.raw_tree),
  };
}

PCGBinarySPDecomposition make_pcg_leaf_node(parallel_layer_guid_t const &l) {
  return PCGBinarySPDecomposition{
      make_generic_binary_sp_leaf(l),
  };
}

PCGBinarySeriesSplit require_series(PCGBinarySPDecomposition const &d) {
  return PCGBinarySeriesSplit{
    require_series(d.raw_tree),
  };
}

PCGBinaryParallelSplit require_parallel(PCGBinarySPDecomposition const &d) {
  return PCGBinaryParallelSplit{
    require_parallel(d.raw_tree),
  };
}

parallel_layer_guid_t require_leaf(PCGBinarySPDecomposition const &d) {
  return require_leaf(d.raw_tree);
}

} // namespace FlexFlow
