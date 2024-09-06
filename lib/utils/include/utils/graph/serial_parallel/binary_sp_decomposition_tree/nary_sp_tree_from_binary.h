#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_NARY_SP_TREE_FROM_BINARY_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_NARY_SP_TREE_FROM_BINARY_H

#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.dtg.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"

namespace FlexFlow {

SerialParallelDecomposition
    nary_sp_tree_from_binary(BinarySPDecompositionTree const &);

} // namespace FlexFlow

#endif
