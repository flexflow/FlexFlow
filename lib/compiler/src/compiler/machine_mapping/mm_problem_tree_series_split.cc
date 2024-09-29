#include "compiler/machine_mapping/mm_problem_tree_series_split.h"
#include "compiler/machine_mapping/full_binary_tree/require.h"

namespace FlexFlow {

MachineMappingProblemTree const &get_left_child(MMProblemTreeSeriesSplit const &s) {
  FullBinaryTree< require_parent(s.problem_tree.raw_tree);
}

MachineMappingProblemTree const &get_right_child(MMProblemTreeSeriesSplit const &) {

}

AbstractedTensorSetMovement const &get_abstracted_tensor_movement(MMProblemTreeSeriesSplit const &) {

}

} // namespace FlexFlow
