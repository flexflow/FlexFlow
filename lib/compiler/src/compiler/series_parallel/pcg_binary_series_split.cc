#include "compiler/series_parallel/pcg_binary_series_split.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_left_child.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_right_child.h"

namespace FlexFlow {

PCGBinarySPDecomposition get_left_child(PCGBinarySeriesSplit const &s) {
  return PCGBinarySPDecomposition{
    get_left_child(s.raw_split),
  };
}

PCGBinarySPDecomposition get_right_child(PCGBinarySeriesSplit const &s) {
  return PCGBinarySPDecomposition{
    get_right_child(s.raw_split),
  };
}

} // namespace FlexFlow
