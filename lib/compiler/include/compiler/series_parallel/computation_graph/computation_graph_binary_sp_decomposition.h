#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_SERIES_PARALLEL_COMPUTATION_GRAPH_BINARY_SP_DECOMPOSITION_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_SERIES_PARALLEL_COMPUTATION_GRAPH_BINARY_SP_DECOMPOSITION_H

#include "compiler/series_parallel/computation_graph/computation_graph_binary_sp_decomposition.dtg.h"
#include "pcg/computation_graph.dtg.h"
#include "pcg/file_format/v1/v1_binary_sp_decomposition/v1_binary_sp_decomposition.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree_implementation.dtg.h"
#include "utils/graph/series_parallel/sp_decomposition_tree_node_type.dtg.h"
#include "utils/overload.h"

namespace FlexFlow {

GenericBinarySPDecompositionTreeImplementation<
    ComputationGraphBinarySPDecomposition,
    ComputationGraphBinarySeriesSplit,
    ComputationGraphBinaryParallelSplit,
    layer_guid_t>
    generic_impl_for_computation_graph_sp_tree();

SPDecompositionTreeNodeType
    get_node_type(ComputationGraphBinarySPDecomposition const &);

ComputationGraphBinarySPDecomposition
    computation_graph_sp_decomp_from_binary_sp_decomp(
        BinarySPDecompositionTree const &);

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

V1BinarySPDecomposition to_v1(ComputationGraphBinarySPDecomposition const &,
                              bidict<int, layer_guid_t> const &layer_numbering);

} // namespace FlexFlow

#endif
