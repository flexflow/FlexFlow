#include "compiler/series_parallel/pcg/pcg_binary_series_split.h"
#include "compiler/series_parallel/pcg/pcg_binary_sp_decomposition.h"

namespace FlexFlow {

BinarySeriesSplit binary_series_split_from_pcg_series_split(
    PCGBinarySeriesSplit const &pcg_split) {
  return BinarySeriesSplit{
      binary_sp_tree_from_pcg_sp_tree(pcg_split.get_left_child()),
      binary_sp_tree_from_pcg_sp_tree(pcg_split.get_right_child()),
  };
}

} // namespace FlexFlow
