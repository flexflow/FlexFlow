#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_VISIT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_VISIT_H

#include "utils/exception.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree.dtg.h"

namespace FlexFlow {

template <typename Result, typename F, typename SeriesLabel, typename ParallelLabel, typename LeafLabel>
Result visit(GenericBinarySPDecompositionTree<SeriesLabel, ParallelLabel, LeafLabel> const &tt, F f) {
  SPDecompositionTreeNodeType node_type = get_node_type(tt);
  switch (node_type) {
    case SPDecompositionTreeNodeType::SERIES: {
      Result result = f(require_series_split(t));
      return result;
    }
    case SPDecompositionTreeNodeType::PARALLEL: {
      Result result = f(require_parallel_split(t));
      return result;
    }
    case SPDecompositionTreeNodeType::NODE: {
      Result result = f(require_leaf(t));
      return result;
    }
    default:
      throw mk_runtime_error(fmt::format("Unknown SPDecompositionTreeNodeType: {}", node_type));
  }
}

} // namespace FlexFlow

#endif
