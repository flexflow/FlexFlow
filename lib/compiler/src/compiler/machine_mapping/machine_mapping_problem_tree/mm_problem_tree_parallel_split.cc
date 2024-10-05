#include "compiler/machine_mapping/machine_mapping_problem_tree/mm_problem_tree_parallel_split.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_left_child.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_right_child.h"

namespace FlexFlow {

MachineMappingProblemTree get_lhs_child(MMProblemTreeParallelSplit const &p) {
  return MachineMappingProblemTree{
      get_left_child(p.raw_split),
  };
}

MachineMappingProblemTree get_rhs_child(MMProblemTreeParallelSplit const &p) {
  return MachineMappingProblemTree{
      get_right_child(p.raw_split),
  };
}

} // namespace FlexFlow
