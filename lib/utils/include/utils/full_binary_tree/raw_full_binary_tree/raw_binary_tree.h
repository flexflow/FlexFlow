#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_RAW_FULL_BINARY_TREE_RAW_BINARY_TREE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_RAW_FULL_BINARY_TREE_RAW_BINARY_TREE_H

#include <memory>
#include <tuple>
#include "utils/full_binary_tree/raw_full_binary_tree/any_value_type.h"
#include <optional>

namespace FlexFlow {

struct RawBinaryTree {
  explicit RawBinaryTree(
    any_value_type const &label,
    RawBinaryTree const &lhs,
    RawBinaryTree const &rhs);
  explicit RawBinaryTree(
    any_value_type const &label);

  RawBinaryTree(RawBinaryTree const &) = default;

  bool operator==(RawBinaryTree const &) const;
  bool operator!=(RawBinaryTree const &) const;

  RawBinaryTree const &left_child() const;
  RawBinaryTree const &right_child() const;

  bool is_leaf() const;
public:
  any_value_type label;
  std::shared_ptr<RawBinaryTree> left_child_ptr;
  std::shared_ptr<RawBinaryTree> right_child_ptr;
private:
  std::tuple<any_value_type,
             std::optional<RawBinaryTree>,
             std::optional<RawBinaryTree>>
    value_tie() const;
  std::tuple<any_value_type const &,
             std::shared_ptr<RawBinaryTree> const &,
             std::shared_ptr<RawBinaryTree> const &>
    ptr_tie() const;

  friend std::hash<RawBinaryTree>;
};

std::string format_as(RawBinaryTree const &);
std::ostream &operator<<(std::ostream &, RawBinaryTree const &);

RawBinaryTree raw_binary_tree_make_leaf(any_value_type const &label);
RawBinaryTree raw_binary_tree_make_parent(any_value_type const &label, RawBinaryTree const &lhs, RawBinaryTree const &rhs);

} // namespace FlexFlow

namespace std {

template <>
struct hash<::FlexFlow::RawBinaryTree> {
  size_t operator()(::FlexFlow::RawBinaryTree const &) const;
};

}

#endif
