#include "compiler/series_parallel/computation_graph_binary_sp_decomposition.h"
#include "compiler/series_parallel/get_computation_graph_series_parallel_decomposition.h"
#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/get_nodes.h"
#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/is_binary_sp_tree_left_associative.h"
#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/is_binary_sp_tree_right_associative.h"
#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/left_associative_binary_sp_tree_from_nary.h"
#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/right_associative_binary_sp_tree_from_nary.h"
#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree_functions.h"

namespace FlexFlow {

SPDecompositionTreeNodeType get_node_type(ComputationGraphBinarySPDecomposition const &d) {
  return d.raw_tree.get_node_type();
}

ComputationGraphBinarySPDecomposition get_left_child(ComputationGraphBinarySPDecomposition const &d) { 
  return ComputationGraphBinarySPDecomposition{
    get_left_child(d.raw_tree),
  };
}

ComputationGraphBinarySPDecomposition get_right_child(ComputationGraphBinarySPDecomposition const &d) {
  return ComputationGraphBinarySPDecomposition{
    get_right_child(d.raw_tree),
  };
}

layer_guid_t require_node(ComputationGraphBinarySPDecomposition const &d) {
  return d.raw_tree.require_node();
}

std::optional<ComputationGraphBinarySPDecomposition> get_computation_graph_left_assoc_binary_sp_decomposition(ComputationGraph const &cg) {
  SerialParallelDecomposition sp_decomposition = ({
    std::optional<SerialParallelDecomposition> result = get_computation_graph_series_parallel_decomposition(cg);
    if (!result.has_value()) {
      return std::nullopt;
    }
    result.value();
  });

  BinarySPDecompositionTree raw_binary_tree = left_associative_binary_sp_tree_from_nary(sp_decomposition);

  return ComputationGraphBinarySPDecomposition{
    transform(raw_binary_tree.raw_tree, [](Node const &n) { return layer_guid_t{n}; })
  };
}

std::optional<ComputationGraphBinarySPDecomposition> get_computation_graph_right_assoc_binary_sp_decomposition(ComputationGraph const &cg) {
  SerialParallelDecomposition sp_decomposition = ({
    std::optional<SerialParallelDecomposition> result = get_computation_graph_series_parallel_decomposition(cg);
    if (!result.has_value()) {
      return std::nullopt;
    }
    result.value();
  });

  BinarySPDecompositionTree raw_binary_tree = right_associative_binary_sp_tree_from_nary(sp_decomposition);

  return ComputationGraphBinarySPDecomposition{
    transform(raw_binary_tree.raw_tree, [](Node const &n) { return layer_guid_t{n}; })
  };
}

bool is_left_associative(ComputationGraphBinarySPDecomposition const &d) {
  return is_binary_sp_tree_left_associative(d.raw_tree);
}

bool is_right_associative(ComputationGraphBinarySPDecomposition const &d) {
  return is_binary_sp_tree_right_associative(d.raw_tree);
}

std::unordered_multiset<layer_guid_t> get_layers(ComputationGraphBinarySPDecomposition const &d) {
  return get_nodes(d.raw_tree);
}

} // namespace FlexFlow
