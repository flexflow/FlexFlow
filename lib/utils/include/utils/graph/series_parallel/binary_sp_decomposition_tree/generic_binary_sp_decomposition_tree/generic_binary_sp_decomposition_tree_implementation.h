#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_IMPLEMENTATION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_IMPLEMENTATION_H

#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree_implementation.dtg.h"
#include "utils/full_binary_tree/full_binary_tree_implementation.dtg.h"
#include <variant>
#include "utils/overload.h"
#include "utils/exception.h"

namespace FlexFlow {

template <typename Tree, typename Series, typename Parallel, typename Leaf>
FullBinaryTreeImplementation<Tree, std::variant<Series, Parallel>, Leaf> 
  get_full_binary_impl_from_generic_sp_impl(GenericBinarySPDecompositionTreeImplementation<Tree, Series, Parallel, Leaf> const &impl) {

  using Parent = std::variant<Series, Parallel>;

  auto full_binary_impl = FullBinaryTreeImplementation<Tree, Parent, Leaf>{
    /*get_left_child=*/[impl](Parent const &parent) -> Tree const & {
      return std::visit(overload {
        [&](Series const &series) -> Tree const & {
          return impl.series_get_left_child(series);
        },
        [&](Parallel const &parallel) -> Tree const & {
          return impl.parallel_get_left_child(parallel);
        },
      }, parent);
    },
    /*get_right_child=*/[impl](Parent const &parent) -> Tree const & {
      return std::visit(overload {
        [&](Series const &series) -> Tree const & {
          return impl.series_get_right_child(series);
        },
        [&](Parallel const &parallel) -> Tree const & {
          return impl.parallel_get_right_child(parallel);
        },
      }, parent);
    },
    /*is_leaf=*/[impl](Tree const &tree) -> bool {
      return impl.get_node_type(tree) == SPDecompositionTreeNodeType::NODE;
    },
    /*require_leaf=*/[impl](Tree const &tree) -> Leaf const & {
      return impl.require_leaf(tree);
    },
    /*require_parent=*/[impl](Tree const &tree) -> Parent {
      SPDecompositionTreeNodeType node_type = impl.get_node_type(tree);
      switch (node_type) {
        case SPDecompositionTreeNodeType::SERIES:
          return Parent{impl.require_series(tree)};
        case SPDecompositionTreeNodeType::PARALLEL:
          return Parent{impl.require_parallel(tree)};
        default:
          throw mk_runtime_error(fmt::format("Unexpected SPDecompositionTreeNodeType: {}", node_type));
      }
    }
  };

  return full_binary_impl;
}

} // namespace FlexFlow

#endif
