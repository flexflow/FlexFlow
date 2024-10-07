#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_CHILD_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_CHILD_H

#include "utils/exception.h"
#include "utils/full_binary_tree/binary_tree_path_entry.dtg.h"
#include "utils/full_binary_tree/full_binary_tree_implementation.dtg.h"
#include <fmt/format.h>

namespace FlexFlow {

template <typename Tree, typename Parent, typename Leaf>
Tree get_child(Parent const &parent, 
               FullBinaryTreeImplementation<Tree, Parent, Leaf> const &impl,
               BinaryTreePathEntry const &e) {
  switch (e) {
    case BinaryTreePathEntry::LEFT_CHILD:
      return impl.get_left_child(parent);
    case BinaryTreePathEntry::RIGHT_CHILD:
      return impl.get_right_child(parent);
    default:
      throw mk_runtime_error(
          fmt::format("Unhandled BinaryTreePathEntry value: {}", e));
  }
}

} // namespace FlexFlow

#endif
