#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_leaves.h"

namespace FlexFlow {

template 
  std::unordered_multiset<int>
      get_leaves(GenericBinarySPDecompositionTree<int, int, int> const &);
template
  std::unordered_multiset<int> get_leaves(GenericBinarySeriesSplit<int, int, int> const &);
template
  std::unordered_multiset<int> get_leaves(GenericBinaryParallelSplit<int, int, int> const &);

} // namespace FlexFlow
