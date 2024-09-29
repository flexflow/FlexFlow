#include "compiler/machine_mapping/mm_problem_tree_split_label.h"
#include "utils/overload.h"

namespace FlexFlow {

SPDecompositionTreeNodeType split_label_get_node_type(MMProblemTreeSplitLabel const &l) {
  return l.visit<SPDecompositionTreeNodeType>(overload {
    [](MMProblemTreeSeriesSplitLabel const &) {
      return SPDecompositionTreeNodeType::SERIES;
    },
    [](MMProblemTreeParallelSplitLabel const &) {
      return SPDecompositionTreeNodeType::PARALLEL;
    },
  });
}

} // namespace FlexFlow
