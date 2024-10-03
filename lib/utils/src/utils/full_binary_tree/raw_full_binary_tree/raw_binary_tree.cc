#include "utils/full_binary_tree/raw_full_binary_tree/raw_binary_tree.h"
#include "utils/hash-utils.h"
#include "utils/hash/tuple.h"

namespace FlexFlow {

RawBinaryTree::RawBinaryTree(
  any_value_type const &label,
  RawBinaryTree const &lhs,
  RawBinaryTree const &rhs) 
  : label(label), 
    left_child_ptr(std::make_shared<RawBinaryTree>(lhs)), 
    right_child_ptr(std::make_shared<RawBinaryTree>(rhs))
{ }

RawBinaryTree::RawBinaryTree(
    any_value_type const &label)
  : label(label), left_child_ptr(nullptr), right_child_ptr(nullptr)
{ }

bool RawBinaryTree::operator==(RawBinaryTree const &other) const {
  if (this->ptr_tie() == other.ptr_tie()) {
    return true;
  }

  return (this->value_tie() == other.value_tie());
}

bool RawBinaryTree::operator!=(RawBinaryTree const &other) const {
  if (this->ptr_tie() == other.ptr_tie()) {
    return false;
  }

  return (this->value_tie() != other.value_tie());
}

RawBinaryTree const &RawBinaryTree::left_child() const {
  return *this->left_child_ptr;
}

RawBinaryTree const &RawBinaryTree::right_child() const {
  return *this->right_child_ptr;
}

bool RawBinaryTree::is_leaf() const {
  return this->left_child_ptr == nullptr && this->right_child_ptr == nullptr;
}

std::tuple<any_value_type,
           std::optional<RawBinaryTree>,
           std::optional<RawBinaryTree>>
  RawBinaryTree::value_tie() const {

  auto ptr_to_optional = [](std::shared_ptr<RawBinaryTree> const &ptr) 
    -> std::optional<RawBinaryTree> {
    if (ptr == nullptr) {
      return std::nullopt;
    } else {
      return *ptr;
    }
  };

  return {this->label, ptr_to_optional(this->left_child_ptr), ptr_to_optional(this->right_child_ptr)};
}

std::tuple<any_value_type const &,
           std::shared_ptr<RawBinaryTree> const &,
           std::shared_ptr<RawBinaryTree> const &>
  RawBinaryTree::ptr_tie() const {
  return std::tie(this->label, this->left_child_ptr, this->right_child_ptr);
}

std::string format_as(RawBinaryTree const &t) {
  if (t.is_leaf()) {
    return fmt::to_string(t.label);
  } else {
    return fmt::format("({} {} {})", t.label, t.left_child(), t.right_child());
  }
}

std::ostream &operator<<(std::ostream &s, RawBinaryTree const &t) {
  return (s << fmt::to_string(t));
}

RawBinaryTree raw_binary_tree_make_leaf(any_value_type const &label) {
  return RawBinaryTree{label};
}

RawBinaryTree raw_binary_tree_make_parent(any_value_type const &label, RawBinaryTree const &lhs, RawBinaryTree const &rhs) {
  return RawBinaryTree{label, lhs, rhs};
}

} // namespace FlexFlow

namespace std {

size_t hash<::FlexFlow::RawBinaryTree>::operator()(::FlexFlow::RawBinaryTree const &t) const {
  return ::FlexFlow::get_std_hash(t.value_tie());
}

} // namespace std
