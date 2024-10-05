#include "compiler/machine_mapping/machine_mapping_problem_tree/mm_problem_tree_series_split.h"

namespace FlexFlow {

MachineMappingProblemTree get_pre_child(MMProblemTreeSeriesSplit const &s) {
  return MachineMappingProblemTree{
      s.raw_split.pre,
  };
}

MachineMappingProblemTree get_post_child(MMProblemTreeSeriesSplit const &s) {
  return MachineMappingProblemTree{
      s.raw_split.post,
  };
}

AbstractedTensorSetMovement const &
    get_abstracted_tensor_movement(MMProblemTreeSeriesSplit const &s) {
  return s.raw_split.label.tensor_set_movement;
}

} // namespace FlexFlow
