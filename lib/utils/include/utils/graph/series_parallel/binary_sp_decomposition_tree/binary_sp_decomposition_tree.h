#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_BINARY_SP_DECOMPOSITION_TREE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_BINARY_SP_DECOMPOSITION_TREE_H

#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_series_split.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_parallel_split.dtg.h"
#include "utils/graph/series_parallel/sp_decomposition_tree_node_type.dtg.h"
#include <unordered_set>

namespace FlexFlow {

BinarySPDecompositionTree make_series_split(BinarySPDecompositionTree const &,
                                            BinarySPDecompositionTree const &);
BinarySPDecompositionTree
    make_parallel_split(BinarySPDecompositionTree const &,
                        BinarySPDecompositionTree const &);
BinarySPDecompositionTree make_leaf_node(Node const &);

bool is_binary_sp_tree_left_associative(BinarySPDecompositionTree const &);
bool is_binary_sp_tree_right_associative(BinarySPDecompositionTree const &);

std::unordered_multiset<Node> get_leaves(BinarySPDecompositionTree const &);

BinarySeriesSplit require_series(BinarySPDecompositionTree const &);
BinaryParallelSplit require_parallel(BinarySPDecompositionTree const &);
Node require_leaf(BinarySPDecompositionTree const &);

SPDecompositionTreeNodeType get_node_type(BinarySPDecompositionTree const &);

template <typename Return, typename F>
Return visit(BinarySPDecompositionTree const &tree, F &&f) {
  SPDecompositionTreeNodeType node_type = get_node_type(tree);
  switch (node_type) {
    case SPDecompositionTreeNodeType::SERIES: {
      Return result = f(require_series(tree));
      return result;
    }
    case SPDecompositionTreeNodeType::PARALLEL: {
      Return result = f(require_parallel(tree));
      return result;
    }
    case SPDecompositionTreeNodeType::NODE: {
      Return result = f(require_leaf(tree));
      return result;
    }
    default:
      throw mk_runtime_error(fmt::format("Unhandled SPDecompositionTreeNodeType value: {}", node_type));
  }
}

} // namespace FlexFlow

#endif
