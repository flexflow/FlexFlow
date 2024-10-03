#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/require.h"

namespace FlexFlow {

template 
  GenericBinarySeriesSplit<int, int, int>
      require_generic_binary_series_split(GenericBinarySPDecompositionTree<int, int, int> const &);
template 
  GenericBinaryParallelSplit<int, int, int>
      require_generic_binary_parallel_split(GenericBinarySPDecompositionTree<int, int, int> const &);
template 
  int require_generic_binary_leaf(GenericBinarySPDecompositionTree<int, int, int> const &);

} // namespace FlexFlow
