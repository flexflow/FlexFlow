#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_BINARY_TREE_PATH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_BINARY_TREE_PATH_H

#include "utils/full_binary_tree/binary_tree_path.dtg.h"

namespace FlexFlow {

BinaryTreePath binary_tree_root_path();
BinaryTreePath nest_inside_left_child(BinaryTreePath const &);
BinaryTreePath nest_inside_right_child(BinaryTreePath const &);

BinaryTreePathEntry binary_tree_path_get_top_level(BinaryTreePath const &);
BinaryTreePath binary_tree_path_get_non_top_level(BinaryTreePath const &);

} // namespace FlexFlow

#endif
