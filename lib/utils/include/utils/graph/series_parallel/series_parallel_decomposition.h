#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_SERIES_PARALLEL_DECOMPOSITION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_SERIES_PARALLEL_DECOMPOSITION_H

#include "utils/graph/series_parallel/intermediate_sp_decomposition_tree.dtg.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.dtg.h"
#include <variant>

namespace FlexFlow {

std::variant<SeriesSplit, ParallelSplit, Node> internal_to_final_ast(
    std::variant<IntermediateSpDecompositionTree, Node> const &ast);
SeriesParallelDecomposition
    to_final_ast(std::variant<IntermediateSpDecompositionTree, Node> const &);

std::unordered_multiset<Node> get_nodes(SeriesParallelDecomposition const &sp);
std::unordered_multiset<Node> get_nodes(SeriesSplit const &);
std::unordered_multiset<Node> get_nodes(ParallelSplit const &);
std::unordered_multiset<Node> get_nodes(Node const &);

} // namespace FlexFlow

#endif
