#ifndef _FLEXFLOW_UTILS_GRAPH_SERIALPARALLEL_INTERNAL_H
#define _FLEXFLOW_UTILS_GRAPH_SERIALPARALLEL_INTERNAL_H

#include "utils/graph/digraph/digraph_view.h"
#include "utils/graph/serial_parallel/serialparallel.h"
#include "utils/graph/serial_parallel/intermediate_sp_decomposition_tree.dtg.h"
#include "utils/visitable.h"
#include <variant>
#include <vector>

namespace FlexFlow {

struct ParallelInternal;

std::variant<IntermediateSpDecompositionTree, Node> sp_decomposition(DiGraphView const &g);
IntermediateSpDecompositionTree parallel_decomposition(DiGraphView const &g);

std::unordered_set<Node>
    from_source_to_sink(DiGraphView const &, Node const &src, Node const &sink);

std::variant<Serial, Parallel, Node> internal_to_final_ast(std::variant<IntermediateSpDecompositionTree, Node> const &);
SerialParallelDecomposition to_final_ast(std::variant<IntermediateSpDecompositionTree, Node> const &);
std::variant<IntermediateSpDecompositionTree, Node> flatten_ast(std::variant<IntermediateSpDecompositionTree, Node> const &ast);

} // namespace FlexFlow

#endif
