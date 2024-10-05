#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_right_child.h"

namespace FlexFlow {

template GenericBinarySPDecompositionTree<int, int, int>
    get_right_child(GenericBinarySeriesSplit<int, int, int> const &);
template GenericBinarySPDecompositionTree<int, int, int>
    get_right_child(GenericBinaryParallelSplit<int, int, int> const &);

} // namespace FlexFlow
