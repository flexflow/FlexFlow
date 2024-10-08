#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_VISIT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_VISIT_H

#include "utils/exception.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree_implementation.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree_visitor.dtg.h"

namespace FlexFlow {

template <typename ReturnType,
          typename Tree,
          typename Series,
          typename Parallel,
          typename Leaf>
ReturnType
    visit(Tree const &tree,
          GenericBinarySPDecompositionTreeImplementation<Tree,
                                                         Series,
                                                         Parallel,
                                                         Leaf> const &impl,
          GenericBinarySPDecompositionTreeVisitor<ReturnType,
                                                  Tree,
                                                  Series,
                                                  Parallel,
                                                  Leaf> const &visitor) {
  SPDecompositionTreeNodeType node_type = impl.get_node_type(tree);
  switch (node_type) {
    case SPDecompositionTreeNodeType::SERIES: {
      ReturnType result = visitor.series_func(impl.require_series(tree));
      return result;
    }
    case SPDecompositionTreeNodeType::PARALLEL: {
      ReturnType result = visitor.parallel_func(impl.require_parallel(tree));
      return result;
    }
    case SPDecompositionTreeNodeType::NODE: {
      ReturnType result = visitor.leaf_func(impl.require_leaf(tree));
      return result;
    }
    default:
      throw mk_runtime_error(fmt::format(
          "Unknown SPDecompositionTreeNodeType value: {}", node_type));
  }
}

} // namespace FlexFlow

#endif
