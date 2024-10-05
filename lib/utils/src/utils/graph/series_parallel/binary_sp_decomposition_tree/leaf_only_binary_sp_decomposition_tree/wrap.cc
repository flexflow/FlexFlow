#include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/wrap.h"

namespace FlexFlow {

template LeafOnlyBinarySPDecompositionTree<int>
    wrap_series_split(LeafOnlyBinarySeriesSplit<int> const &);
template LeafOnlyBinarySPDecompositionTree<int>
    wrap_parallel_split(LeafOnlyBinaryParallelSplit<int> const &);

} // namespace FlexFlow
