#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_series_split.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_left_child.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_right_child.h"

namespace FlexFlow {

BinarySPDecompositionTree get_left_child(BinarySeriesSplit const &split) {
  return BinarySPDecompositionTree{
    get_left_child(split.raw_split),
  };
}

BinarySPDecompositionTree get_right_child(BinarySeriesSplit const &split) {
  return BinarySPDecompositionTree{
    get_right_child(split.raw_split),
  };
}

} // namespace FlexFlow
