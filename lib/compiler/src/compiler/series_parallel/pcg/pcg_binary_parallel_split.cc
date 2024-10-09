#include "compiler/series_parallel/pcg/pcg_binary_parallel_split.h"
#include "compiler/series_parallel/pcg/pcg_binary_sp_decomposition.h"

namespace FlexFlow {

BinaryParallelSplit binary_parallel_split_from_pcg_parallel_split(
    PCGBinaryParallelSplit const &pcg_split) {
  return BinaryParallelSplit{
      binary_sp_tree_from_pcg_sp_tree(pcg_split.get_left_child()),
      binary_sp_tree_from_pcg_sp_tree(pcg_split.get_right_child()),
  };
}

} // namespace FlexFlow
