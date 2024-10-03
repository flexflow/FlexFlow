#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_GET_NODE_TYPE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_GET_NODE_TYPE_H

#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_split_label.h"
#include "utils/graph/series_parallel/sp_decomposition_tree_node_type.dtg.h"
#include "utils/full_binary_tree/visit.h"
#include "utils/overload.h"

namespace FlexFlow {

template <typename SeriesLabel, typename ParallelLabel, typename LeafLabel>
SPDecompositionTreeNodeType
    get_node_type(GenericBinarySPDecompositionTree<SeriesLabel, ParallelLabel, LeafLabel> const &tt) {
  auto visitor = FullBinaryTreeVisitor<SPDecompositionTreeNodeType, GenericBinarySPSplitLabel<SeriesLabel, ParallelLabel>, LeafLabel>{
    [](FullBinaryTreeParentNode<GenericBinarySPSplitLabel<SeriesLabel, ParallelLabel>, LeafLabel> const &parent) {
      return get_node_type(get_full_binary_tree_parent_label(parent));
    },
    [](LeafLabel const &) {
      return SPDecompositionTreeNodeType::NODE;
    },
  };

  return visit(tt.raw_tree, visitor);
}

} // namespace FlexFlow

#endif
