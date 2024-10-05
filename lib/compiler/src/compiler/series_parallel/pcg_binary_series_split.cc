#include "compiler/series_parallel/pcg_binary_series_split.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/leaf_only_binary_series_split.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/transform.h"

namespace FlexFlow {

BinarySeriesSplit get_raw_graph_series_split(PCGBinarySeriesSplit const &s) {
  auto visitor =
      LeafOnlyBinarySPDecompositionTreeVisitor<parallel_layer_guid_t, Node>{
          [](parallel_layer_guid_t const &l) { return l.raw_graph_node; }};

  return BinarySeriesSplit{
      transform(s.raw_split, visitor),
  };
}

PCGBinarySPDecomposition get_left_child(PCGBinarySeriesSplit const &s) {
  return PCGBinarySPDecomposition{
      get_left_child(s.raw_split),
  };
}

PCGBinarySPDecomposition get_right_child(PCGBinarySeriesSplit const &s) {
  return PCGBinarySPDecomposition{
      get_right_child(s.raw_split),
  };
}

} // namespace FlexFlow
