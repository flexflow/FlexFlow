#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GET_NUM_TREE_NODES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GET_NUM_TREE_NODES_H

#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.h"

namespace FlexFlow {

int get_num_tree_nodes(BinarySPDecompositionTree const &);
int get_num_tree_nodes(BinarySeriesSplit const &);
int get_num_tree_nodes(BinaryParallelSplit const &);

} // namespace FlexFlow

#endif
