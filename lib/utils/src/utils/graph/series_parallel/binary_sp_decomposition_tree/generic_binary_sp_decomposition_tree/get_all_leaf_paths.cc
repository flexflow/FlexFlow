#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_all_leaf_paths.h"

namespace FlexFlow {

template
  std::unordered_set<BinaryTreePath> get_all_leaf_paths(GenericBinarySPDecompositionTree<int, int, int> const &);

} // namespace FlexFlow
