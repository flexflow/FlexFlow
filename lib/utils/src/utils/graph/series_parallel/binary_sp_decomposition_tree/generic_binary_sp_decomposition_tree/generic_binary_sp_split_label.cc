#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_split_label.h"

namespace FlexFlow {

template 
  SPDecompositionTreeNodeType get_node_type(GenericBinarySPSplitLabel<int, int> const &);
template 
  GenericBinarySPSplitLabel<int, int> make_generic_binary_series_split_label(int const &);
template
  GenericBinarySPSplitLabel<int, int> make_generic_binary_parallel_split_label(int const &);
template
  int require_generic_binary_series_split_label(GenericBinarySPSplitLabel<int, int> const &);
template
  int require_generic_binary_parallel_split_label(GenericBinarySPSplitLabel<int, int> const &);

} // namespace FlexFlow
