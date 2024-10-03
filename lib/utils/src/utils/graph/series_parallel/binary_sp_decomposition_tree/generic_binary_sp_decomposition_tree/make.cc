#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/make.h"

namespace FlexFlow {

template
  GenericBinarySPDecompositionTree<int, int, int> make_generic_binary_series_split(
    int const &,
    GenericBinarySPDecompositionTree<int, int, int> const &,
    GenericBinarySPDecompositionTree<int, int, int> const &);
template
  GenericBinarySPDecompositionTree<int, int, int> make_generic_binary_parallel_split(
      int const &label,
      GenericBinarySPDecompositionTree<int, int, int> const &,
      GenericBinarySPDecompositionTree<int, int, int> const &);
template
  GenericBinarySPDecompositionTree<int, int, int> make_generic_binary_sp_leaf(int const &);

} // namespace FlexFlow
