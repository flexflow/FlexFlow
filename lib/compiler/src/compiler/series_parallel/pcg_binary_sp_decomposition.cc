#include "compiler/series_parallel/pcg_binary_sp_decomposition.h"

namespace FlexFlow {

std::optional<PCGBinarySPDecomposition>
  get_pcg_balanced_binary_sp_decomposition(ParallelComputationGraph const &) {
  NOT_IMPLEMENTED();
}

std::unordered_multiset<parallel_layer_guid_t>
  get_parallel_layers(PCGBinarySPDecomposition const &) {
  NOT_IMPLEMENTED();
}

SPDecompositionTreeNodeType get_node_type(PCGBinarySPDecomposition const &) {
  NOT_IMPLEMENTED();
}

PCGBinarySPDecomposition make_pcg_series_split(PCGBinarySPDecomposition const &, PCGBinarySPDecomposition const &) {
  NOT_IMPLEMENTED();
}

PCGBinarySPDecomposition make_pcg_parallel_split(PCGBinarySPDecomposition const &, PCGBinarySPDecomposition const &) {
  NOT_IMPLEMENTED();
}

PCGBinarySPDecomposition make_pcg_leaf_node(parallel_layer_guid_t const &) {
  NOT_IMPLEMENTED();
}

PCGBinarySeriesSplit require_series(PCGBinarySPDecomposition const &) {
  NOT_IMPLEMENTED();
}

PCGBinaryParallelSplit require_parallel(PCGBinarySPDecomposition const &) {
  NOT_IMPLEMENTED();
}

parallel_layer_guid_t require_leaf(PCGBinarySPDecomposition const &) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
