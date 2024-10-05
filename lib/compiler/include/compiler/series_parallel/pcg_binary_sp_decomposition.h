#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_SERIES_PARALLEL_PCG_BINARY_SP_DECOMPOSITION_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_SERIES_PARALLEL_PCG_BINARY_SP_DECOMPOSITION_H

#include "compiler/series_parallel/pcg_binary_parallel_split.dtg.h"
#include "compiler/series_parallel/pcg_binary_series_split.dtg.h"
#include "compiler/series_parallel/pcg_binary_sp_decomposition.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"
#include "utils/full_binary_tree/binary_tree_path.dtg.h"
#include "utils/graph/series_parallel/sp_decomposition_tree_node_type.dtg.h"
#include <optional>

namespace FlexFlow {

std::optional<PCGBinarySPDecomposition>
    get_pcg_balanced_binary_sp_decomposition(ParallelComputationGraph const &);
std::unordered_multiset<parallel_layer_guid_t>
    get_parallel_layers(PCGBinarySPDecomposition const &);

SPDecompositionTreeNodeType get_node_type(PCGBinarySPDecomposition const &);

PCGBinarySPDecomposition
    make_pcg_series_split(PCGBinarySPDecomposition const &,
                          PCGBinarySPDecomposition const &);
PCGBinarySPDecomposition
    make_pcg_parallel_split(PCGBinarySPDecomposition const &,
                            PCGBinarySPDecomposition const &);
PCGBinarySPDecomposition make_pcg_leaf_node(parallel_layer_guid_t const &);

PCGBinarySPDecomposition wrap_series_split(PCGBinarySeriesSplit const &);
PCGBinarySPDecomposition wrap_parallel_split(PCGBinaryParallelSplit const &);

PCGBinarySeriesSplit require_series(PCGBinarySPDecomposition const &);
PCGBinaryParallelSplit require_parallel(PCGBinarySPDecomposition const &);
parallel_layer_guid_t require_leaf(PCGBinarySPDecomposition const &);

std::unordered_set<BinaryTreePath>
    find_paths_to_leaf(PCGBinarySPDecomposition const &,
                       parallel_layer_guid_t const &);

template <typename ReturnType, typename F>
ReturnType visit(PCGBinarySPDecomposition const &d, F &&f) {
  SPDecompositionTreeNodeType node_type = get_node_type(d);
  switch (node_type) {
    case SPDecompositionTreeNodeType::SERIES:
      return f(require_series(d));
    case SPDecompositionTreeNodeType::PARALLEL:
      return f(require_parallel(d));
    case SPDecompositionTreeNodeType::NODE:
      return f(require_leaf(d));
    default:
      throw mk_runtime_error(fmt::format("Unknown node type {}", node_type));
  }
}

} // namespace FlexFlow

#endif
