#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_H

#include <variant>
#include <tuple>
#include <memory>

namespace FlexFlow {

template <typename T>
struct GenericBinarySPDecompositionTree;

template <typename T>
struct GenericBinarySeriesSplit {
public:
  GenericBinarySeriesSplit() = delete;
  explicit GenericBinarySeriesSplit(
      GenericBinarySPDecompositionTree<T> const &lhs,
      GenericBinarySPDecompositionTree<T> const &rhs)
      : left_child_ptr(
            std::make_shared<GenericBinarySPDecompositionTree<T>>(lhs)),
        right_child_ptr(
            std::make_shared<GenericBinarySPDecompositionTree<T>>(rhs)) {}

  GenericBinarySeriesSplit(GenericBinarySeriesSplit const &) = default;

  bool operator==(GenericBinarySeriesSplit const &other) const {
    return this->tie() == other.tie();
  }

  bool operator!=(GenericBinarySeriesSplit const &other) const {
    return this->tie() != other.tie();
  }

  bool operator<(GenericBinarySeriesSplit const &other) const {
    return this->tie() < other.tie();
  }

public:
  std::shared_ptr<GenericBinarySPDecompositionTree<T>> left_child_ptr;
  std::shared_ptr<GenericBinarySPDecompositionTree<T>> right_child_ptr;

private:
  std::tuple<GenericBinarySPDecompositionTree<T> const &,
             GenericBinarySPDecompositionTree<T> const &>
      tie() const {
    return std::tie(*this->left_child_ptr, *this->right_child_ptr);
  }

  friend std::hash<GenericBinarySeriesSplit>;
};

template <typename T>
struct GenericBinaryParallelSplit {
public:
  GenericBinaryParallelSplit() = delete;
  explicit GenericBinaryParallelSplit(
      GenericBinarySPDecompositionTree<T> const &lhs,
      GenericBinarySPDecompositionTree<T> const &rhs)
      : left_child_ptr(
            std::make_shared<GenericBinarySPDecompositionTree<T>>(lhs)),
        right_child_ptr(
            std::make_shared<GenericBinarySPDecompositionTree<T>>(rhs)) {}

  GenericBinaryParallelSplit(GenericBinaryParallelSplit const &) = default;

  bool operator==(GenericBinaryParallelSplit const &other) const {
    return this->tie() == other.tie();
  }

  bool operator!=(GenericBinaryParallelSplit const &other) const {
    return this->tie() != other.tie();
  }

  bool operator<(GenericBinaryParallelSplit const &other) const {
    return this->tie() < other.tie();
  }

public:
  std::shared_ptr<GenericBinarySPDecompositionTree<T>> left_child_ptr;
  std::shared_ptr<GenericBinarySPDecompositionTree<T>> right_child_ptr;

private:
  std::tuple<GenericBinarySPDecompositionTree<T> const &,
             GenericBinarySPDecompositionTree<T> const &>
      tie() const {
    return std::tie(*this->left_child_ptr, *this->right_child_ptr);
  }

  friend std::hash<GenericBinaryParallelSplit>;
};

template <typename T>
struct GenericBinarySPDecompositionTree {
public:
  GenericBinarySPDecompositionTree() = delete;
  explicit GenericBinarySPDecompositionTree(
      GenericBinarySeriesSplit<T> const &s)
      : root{s} {}

  explicit GenericBinarySPDecompositionTree(
      GenericBinaryParallelSplit<T> const &s)
      : root{s} {}

  explicit GenericBinarySPDecompositionTree(T const &t) : root{t} {}

  GenericBinarySPDecompositionTree(GenericBinarySPDecompositionTree const &) =
      default;

  bool operator==(GenericBinarySPDecompositionTree const &other) const {
    return this->tie() == other.tie();
  }

  bool operator!=(GenericBinarySPDecompositionTree const &other) const {
    return this->tie() != other.tie();
  }

  bool operator<(GenericBinarySPDecompositionTree const &other) const {
    return this->tie() < other.tie();
  }

public:
  std::variant<GenericBinarySeriesSplit<T>, GenericBinaryParallelSplit<T>, T>
      root;

private:
  std::tuple<decltype(root) const &> tie() const {
    return std::tie(this->root);
  }

  friend std::hash<GenericBinarySPDecompositionTree>;
};

} // namespace FlexFlow

// namespace rc {
//
// template <>
// struct Arbitrary<::FlexFlow::BinarySeriesSplit> {
//   static Gen<::FlexFlow::BinarySeriesSplit> arbitrary();
// };
//
// template <>
// struct Arbitrary<::FlexFlow::GenericBinaryParallelSplit> {
//   static Gen<::FlexFlow::GenericBinaryParallelSplit> arbitrary();
// };
//
// template <>
// struct Arbitrary<::FlexFlow::GenericBinarySPDecompositionTree> {
//   static Gen<::FlexFlow::GenericBinarySPDecompositionTree> arbitrary();
// };
//
// } // namespace rc

#endif
