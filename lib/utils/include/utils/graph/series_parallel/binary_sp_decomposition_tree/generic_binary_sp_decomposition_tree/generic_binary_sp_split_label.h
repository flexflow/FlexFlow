#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_SPLIT_LABEL_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_SPLIT_LABEL_H

#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_split_label.dtg.h"
#include "utils/graph/series_parallel/sp_decomposition_tree_node_type.dtg.h"
#include "utils/overload.h"

namespace FlexFlow {

template <typename SeriesLabel, typename ParallelLabel>
SPDecompositionTreeNodeType get_node_type(GenericBinarySPSplitLabel<SeriesLabel, ParallelLabel> const &label) {
  return label.template visit<SPDecompositionTreeNodeType>(overload {
    [](SeriesLabel const &) { return SPDecompositionTreeNodeType::SERIES; },
    [](ParallelLabel const &) { return SPDecompositionTreeNodeType::PARALLEL; },
  });
}

} // namespace FlexFlow

#endif
