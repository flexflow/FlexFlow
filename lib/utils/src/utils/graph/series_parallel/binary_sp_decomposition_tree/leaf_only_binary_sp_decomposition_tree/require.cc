#include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/require.h"

namespace FlexFlow {

template LeafOnlyBinarySeriesSplit<int> require_leaf_only_binary_series_split(
    LeafOnlyBinarySPDecompositionTree<int> const &);
template LeafOnlyBinaryParallelSplit<int>
    require_leaf_only_binary_parallel_split(
        LeafOnlyBinarySPDecompositionTree<int> const &);
template int require_leaf_only_binary_leaf(
    LeafOnlyBinarySPDecompositionTree<int> const &);

} // namespace FlexFlow
