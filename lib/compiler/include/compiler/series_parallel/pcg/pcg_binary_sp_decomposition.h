#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_SERIES_PARALLEL_PCG_BINARY_SP_DECOMPOSITION_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_SERIES_PARALLEL_PCG_BINARY_SP_DECOMPOSITION_H

#include "compiler/series_parallel/pcg/pcg_binary_parallel_split.dtg.h"
#include "compiler/series_parallel/pcg/pcg_binary_series_split.dtg.h"
#include "compiler/series_parallel/pcg/pcg_binary_sp_decomposition.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"
#include "utils/full_binary_tree/binary_tree_path.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree_implementation.dtg.h"
#include "utils/graph/series_parallel/sp_decomposition_tree_node_type.dtg.h"
#include <optional>

namespace FlexFlow {

GenericBinarySPDecompositionTreeImplementation<PCGBinarySPDecomposition,
                                               PCGBinarySeriesSplit,
                                               PCGBinaryParallelSplit,
                                               parallel_layer_guid_t>
    generic_impl_for_pcg_sp_tree();

BinarySPDecompositionTree
    binary_sp_tree_from_pcg_sp_tree(PCGBinarySPDecomposition const &);

std::optional<PCGBinarySPDecomposition>
    get_pcg_balanced_binary_sp_decomposition(ParallelComputationGraph const &);
std::unordered_multiset<parallel_layer_guid_t>
    get_parallel_layers(PCGBinarySPDecomposition const &);

SPDecompositionTreeNodeType get_node_type(PCGBinarySPDecomposition const &);

std::unordered_set<BinaryTreePath>
    find_paths_to_leaf(PCGBinarySPDecomposition const &,
                       parallel_layer_guid_t const &);

} // namespace FlexFlow

#endif
