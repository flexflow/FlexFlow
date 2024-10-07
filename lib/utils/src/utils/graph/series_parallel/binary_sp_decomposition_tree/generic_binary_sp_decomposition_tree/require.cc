#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/require.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using SeriesLabel = value_type<0>;
using ParallelLabel = value_type<1>;
using LeafLabel = value_type<2>;

template
  GenericBinarySeriesSplit<SeriesLabel, ParallelLabel, LeafLabel>
      require_generic_binary_series_split(
          GenericBinarySPDecompositionTree<SeriesLabel,
                                           ParallelLabel,
                                           LeafLabel> const &);
template 
  GenericBinaryParallelSplit<SeriesLabel, ParallelLabel, LeafLabel>
      require_generic_binary_parallel_split(
          GenericBinarySPDecompositionTree<SeriesLabel,
                                           ParallelLabel,
                                           LeafLabel> const &);
template 
  LeafLabel require_generic_binary_leaf(
      GenericBinarySPDecompositionTree<SeriesLabel,
                                       ParallelLabel,
                                       LeafLabel> const &);
} // namespace FlexFlow
