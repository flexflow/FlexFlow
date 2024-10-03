#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_FULL_BINARY_TREE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_FULL_BINARY_TREE_H

#include <memory>
#include <variant>
#include <tuple>

namespace FlexFlow {

template <typename ParentLabel, typename LeafLabel>
struct FullBinaryTree;

template <typename ParentLabel, typename LeafLabel>
struct FullBinaryTreeParentNode {
  explicit FullBinaryTreeParentNode(
    ParentLabel const &label,
    FullBinaryTree<ParentLabel, LeafLabel> const &lhs,
    FullBinaryTree<ParentLabel, LeafLabel> const &rhs)
  : label(label),
    left_child_ptr(
      std::make_shared<FullBinaryTree<ParentLabel, LeafLabel>>(lhs)),
    right_child_ptr(
      std::make_shared<FullBinaryTree<ParentLabel, LeafLabel>>(rhs))
  { }

  FullBinaryTreeParentNode(FullBinaryTreeParentNode const &) = default;

  bool operator==(FullBinaryTreeParentNode const &other) const {
    if (this->tie_ptr() == other.tie_ptr()) {
      return true;
    }

    return this->tie() == other.tie();
  }

  bool operator!=(FullBinaryTreeParentNode const &other) const {
    if (this->tie_ptr() == other.tie_ptr()) {
      return false;
    }

    return this->tie() != other.tie();
  }

  bool operator<(FullBinaryTreeParentNode const &other) const {
    return this->tie() < other.tie();
  }
public:
  ParentLabel label;
  std::shared_ptr<FullBinaryTree<ParentLabel, LeafLabel>> left_child_ptr;
  std::shared_ptr<FullBinaryTree<ParentLabel, LeafLabel>> right_child_ptr;
private:
  std::tuple<ParentLabel const &,
             std::shared_ptr<FullBinaryTree<ParentLabel, LeafLabel>> const &,
             std::shared_ptr<FullBinaryTree<ParentLabel, LeafLabel>> const &>
    tie_ptr() const {
    return std::tie(this->label, this->left_child_ptr, this->right_child_ptr);
  }

  std::tuple<ParentLabel const &,
             FullBinaryTree<ParentLabel, LeafLabel> const &,
             FullBinaryTree<ParentLabel, LeafLabel> const &>
    tie() const {
    return std::tie(this->label, *this->left_child_ptr, *this->right_child_ptr);
  }

  friend std::hash<FullBinaryTreeParentNode>;
};

template <typename ParentLabel, typename LeafLabel>
struct FullBinaryTree {
public:
  FullBinaryTree() = delete;
  explicit FullBinaryTree(FullBinaryTreeParentNode<ParentLabel, LeafLabel> const &t) 
    : root{t} {}

  explicit FullBinaryTree(LeafLabel const &t)
    : root{t} {}

  bool operator==(FullBinaryTree const &other) const {
    return this->tie() == other.tie();
  }

  bool operator!=(FullBinaryTree const &other) const {
    return this->tie() != other.tie();
  }

  bool operator<(FullBinaryTree const &other) const {
    return this->tie() < other.tie();
  }
public:
  std::variant<FullBinaryTreeParentNode<ParentLabel, LeafLabel>, LeafLabel> root;
private:
  std::tuple<decltype(root) const &> tie() const {
    return std::tie(this->root);
  }

  friend std::hash<FullBinaryTree>;
};

} // namespace FlexFlow

#endif
