#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_BINARY_SP_DECOMPOSITION_TREE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_BINARY_SP_DECOMPOSITION_TREE_H

#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.dtg.h"

namespace FlexFlow {

BinarySPDecompositionTree make_series_split(BinarySPDecompositionTree const &,
                                            BinarySPDecompositionTree const &);
BinarySPDecompositionTree
    make_parallel_split(BinarySPDecompositionTree const &,
                        BinarySPDecompositionTree const &);
BinarySPDecompositionTree make_leaf_node(Node const &);

} // namespace FlexFlow

#endif
