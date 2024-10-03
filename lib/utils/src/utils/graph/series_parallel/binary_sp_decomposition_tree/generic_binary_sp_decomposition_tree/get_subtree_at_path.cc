#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_subtree_at_path.h"

namespace FlexFlow {

template
  std::optional<GenericBinarySPDecompositionTree<int, int, int>> 
    get_subtree_at_path(GenericBinarySPDecompositionTree<int, int, int> const &,
                        BinaryTreePath const &);

} // namespace FlexFlow
