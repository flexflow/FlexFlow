#include "compiler/machine_mapping/machine_mapping_result_tree/mm_result_tree_parallel_split.h"

namespace FlexFlow {

float get_cost(MMResultTreeParallelSplit const &p) {
  return p.raw_split.label.cost;
}

BinaryTreePathEntry get_problem_tree_path_entry(MMResultTreeParallelSplit const &p) {
  return p.raw_split.label.problem_tree_path_entry;
}

} // namespace FlexFlow
