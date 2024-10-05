#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/wrap.h"

namespace FlexFlow {

template GenericBinarySPDecompositionTree<int, int, int>
    wrap_series_split(GenericBinarySeriesSplit<int, int, int> const &);
template GenericBinarySPDecompositionTree<int, int, int>
    wrap_parallel_split(GenericBinaryParallelSplit<int, int, int> const &);

} // namespace FlexFlow
