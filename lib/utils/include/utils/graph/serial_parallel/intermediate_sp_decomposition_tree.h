#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_INTERMEDIATE_SP_DECOMPOSITION_TREE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_INTERMEDIATE_SP_DECOMPOSITION_TREE_H

#include "utils/graph/serial_parallel/intermediate_sp_decomposition_tree.dtg.h"

namespace FlexFlow {

std::variant<IntermediateSpDecompositionTree, Node>
    flatten_ast(std::variant<IntermediateSpDecompositionTree, Node> const &ast);

} // namespace FlexFlow

#endif
