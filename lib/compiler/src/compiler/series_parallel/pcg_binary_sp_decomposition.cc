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

} // namespace FlexFlow
