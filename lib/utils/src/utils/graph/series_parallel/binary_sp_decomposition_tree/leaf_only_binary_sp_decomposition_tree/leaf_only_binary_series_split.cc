#include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/leaf_only_binary_series_split.h"

namespace FlexFlow {

template
  LeafOnlyBinarySPDecompositionTree<int> get_left_child(LeafOnlyBinarySeriesSplit<int> const &);
template 
  LeafOnlyBinarySPDecompositionTree<int> get_right_child(LeafOnlyBinarySeriesSplit<int> const &);

} // namespace FlexFlow
