#include "compiler/machine_mapping/machine_mapping_result_tree/mm_result_tree_series_split.h"

namespace FlexFlow {

float get_cost(MMResultTreeSeriesSplit const &s) {
  return s.raw_split.label.cost;
}

BinaryTreePathEntry get_problem_tree_path_entry(MMResultTreeSeriesSplit const &s) {
  return s.raw_split.label.problem_tree_path_entry;
}

} // namespace FlexFlow
