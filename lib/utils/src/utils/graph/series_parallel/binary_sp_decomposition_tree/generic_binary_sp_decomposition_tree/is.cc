#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/is.h"

namespace FlexFlow {

template bool
    is_series_split(GenericBinarySPDecompositionTree<int, int, int> const &);
template bool
    is_parallel_split(GenericBinarySPDecompositionTree<int, int, int> const &);
template bool is_leaf(GenericBinarySPDecompositionTree<int, int, int> const &);

} // namespace FlexFlow
