#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/find_paths_to_leaf.h"

namespace FlexFlow {

template
  std::unordered_set<BinaryTreePath> find_paths_to_leaf(GenericBinarySPDecompositionTree<int, int, int> const &,
                                                        int const &);

} // namespace FlexFlow
