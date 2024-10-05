#include "utils/full_binary_tree/binary_tree_path.h"
#include "utils/containers/subvec.h"

namespace FlexFlow {

BinaryTreePath binary_tree_root_path() {
  return BinaryTreePath{{}};
}

BinaryTreePath nest_inside_left_child(BinaryTreePath const &p) {
  BinaryTreePath result = p;
  result.entries.insert(result.entries.begin(),
                        BinaryTreePathEntry::LEFT_CHILD);
  return result;
}

BinaryTreePath nest_inside_right_child(BinaryTreePath const &p) {
  BinaryTreePath result = p;
  result.entries.insert(result.entries.begin(),
                        BinaryTreePathEntry::RIGHT_CHILD);
  return result;
}

BinaryTreePathEntry binary_tree_path_get_top_level(BinaryTreePath const &p) {
  return p.entries.at(0);
}

BinaryTreePath binary_tree_path_get_non_top_level(BinaryTreePath const &p) {
  return BinaryTreePath{
      subvec(p.entries, 1, std::nullopt),
  };
}

} // namespace FlexFlow
