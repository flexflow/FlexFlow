#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_GET_NODE_TYPE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_GET_NODE_TYPE_H

#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree.h"
#include "utils/graph/serial_parallel/sp_decomposition_tree_node_type.dtg.h"
#include "utils/overload.h"
#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/visit.h"

namespace FlexFlow {

template <typename T>
SPDecompositionTreeNodeType get_node_type(GenericBinarySPDecompositionTree<T> const &tt) {
  return visit<SPDecompositionTreeNodeType>(tt, overload {
    [](GenericBinarySeriesSplit<T> const &) { 
      return SPDecompositionTreeNodeType::SERIES;
    },
    [](GenericBinaryParallelSplit<T> const &) {
      return SPDecompositionTreeNodeType::PARALLEL;
    },
    [](T const &) {
      return SPDecompositionTreeNodeType::NODE;
    },
  });
}

} // namespace FlexFlow

#endif
