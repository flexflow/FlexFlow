#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_SERIES_PARALLEL_COMPUTATION_GRAPH_BINARY_SP_DECOMPOSITION_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_SERIES_PARALLEL_COMPUTATION_GRAPH_BINARY_SP_DECOMPOSITION_H

#include "compiler/series_parallel/computation_graph_binary_sp_decomposition.dtg.h"
#include "pcg/computation_graph.dtg.h"
#include "utils/graph/serial_parallel/sp_decomposition_tree_node_type.dtg.h"

namespace FlexFlow {

SPDecompositionTreeNodeType
    get_node_type(ComputationGraphBinarySPDecomposition const &);
ComputationGraphBinarySPDecomposition
    get_left_child(ComputationGraphBinarySPDecomposition const &);
ComputationGraphBinarySPDecomposition
    get_right_child(ComputationGraphBinarySPDecomposition const &);
layer_guid_t require_node(ComputationGraphBinarySPDecomposition const &);
std::optional<ComputationGraphBinarySPDecomposition>
    get_computation_graph_left_assoc_binary_sp_decomposition(
        ComputationGraph const &);
std::optional<ComputationGraphBinarySPDecomposition>
    get_computation_graph_right_assoc_binary_sp_decomposition(
        ComputationGraph const &);
bool is_left_associative(ComputationGraphBinarySPDecomposition const &);
bool is_right_associative(ComputationGraphBinarySPDecomposition const &);
std::unordered_multiset<layer_guid_t>
    get_layers(ComputationGraphBinarySPDecomposition const &);

} // namespace FlexFlow

#endif
