#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_num_tree_nodes.h"

namespace FlexFlow {

template int
    get_num_tree_nodes(GenericBinarySPDecompositionTree<int, int, int> const &);
template int
    get_num_tree_nodes(GenericBinarySeriesSplit<int, int, int> const &);
template int
    get_num_tree_nodes(GenericBinaryParallelSplit<int, int, int> const &);

} // namespace FlexFlow
