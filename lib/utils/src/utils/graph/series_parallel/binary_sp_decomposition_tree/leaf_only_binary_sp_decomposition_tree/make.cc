#include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/make.h"

namespace FlexFlow {

template LeafOnlyBinarySPDecompositionTree<int>
    leaf_only_binary_sp_tree_make_series_split(
        LeafOnlyBinarySPDecompositionTree<int> const &,
        LeafOnlyBinarySPDecompositionTree<int> const &);
template LeafOnlyBinarySPDecompositionTree<int>
    leaf_only_binary_sp_tree_make_parallel_split(
        LeafOnlyBinarySPDecompositionTree<int> const &,
        LeafOnlyBinarySPDecompositionTree<int> const &);
template LeafOnlyBinarySPDecompositionTree<int>
    leaf_only_binary_sp_tree_make_leaf(int const &);

} // namespace FlexFlow
