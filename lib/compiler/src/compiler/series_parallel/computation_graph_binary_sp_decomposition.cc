#include "compiler/series_parallel/computation_graph_binary_sp_decomposition.h"
#include "compiler/series_parallel/get_computation_graph_series_parallel_decomposition.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/get_leaves.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/get_node_type.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/is_binary_sp_tree_left_associative.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/is_binary_sp_tree_right_associative.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/require.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/transform.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/left_associative_binary_sp_tree_from_nary.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/right_associative_binary_sp_tree_from_nary.h"

namespace FlexFlow {

SPDecompositionTreeNodeType
    get_node_type(ComputationGraphBinarySPDecomposition const &d) {
  return get_node_type(d.raw_tree);
}

layer_guid_t require_node(ComputationGraphBinarySPDecomposition const &d) {
  return require_leaf(d.raw_tree);
}

std::optional<ComputationGraphBinarySPDecomposition>
    get_computation_graph_left_assoc_binary_sp_decomposition(
        ComputationGraph const &cg) {
  SeriesParallelDecomposition sp_decomposition = ({
    std::optional<SeriesParallelDecomposition> result =
        get_computation_graph_series_parallel_decomposition(cg);
    if (!result.has_value()) {
      return std::nullopt;
    }
    result.value();
  });

  BinarySPDecompositionTree raw_binary_tree =
      left_associative_binary_sp_tree_from_nary(sp_decomposition);

  auto visitor = LeafOnlyBinarySPDecompositionTreeVisitor<Node, layer_guid_t>{
    [](Node const &n) { return layer_guid_t{n}; },
  };
  return ComputationGraphBinarySPDecomposition{transform(
      raw_binary_tree.raw_tree, visitor)};
}

std::optional<ComputationGraphBinarySPDecomposition>
    get_computation_graph_right_assoc_binary_sp_decomposition(
        ComputationGraph const &cg) {
  SeriesParallelDecomposition sp_decomposition = ({
    std::optional<SeriesParallelDecomposition> result =
        get_computation_graph_series_parallel_decomposition(cg);
    if (!result.has_value()) {
      return std::nullopt;
    }
    result.value();
  });

  BinarySPDecompositionTree raw_binary_tree =
      right_associative_binary_sp_tree_from_nary(sp_decomposition);

  auto visitor = LeafOnlyBinarySPDecompositionTreeVisitor<Node, layer_guid_t>{
    [](Node const &n) { return layer_guid_t{n}; },
  };
  return ComputationGraphBinarySPDecomposition{transform(
      raw_binary_tree.raw_tree, visitor)};
}

bool is_left_associative(ComputationGraphBinarySPDecomposition const &d) {
  return is_binary_sp_tree_left_associative(d.raw_tree);
}

bool is_right_associative(ComputationGraphBinarySPDecomposition const &d) {
  return is_binary_sp_tree_right_associative(d.raw_tree);
}

std::unordered_multiset<layer_guid_t>
    get_layers(ComputationGraphBinarySPDecomposition const &d) {
  return get_leaves(d.raw_tree);
}

} // namespace FlexFlow
