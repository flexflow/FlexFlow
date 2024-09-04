#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_BINARY_SP_DECOMPOSITION_TREE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_BINARY_SP_DECOMPOSITION_TREE_H

#include "utils/graph/node/node.dtg.h"
#include "utils/graph/serial_parallel/sp_decomposition_tree_node_type.dtg.h"
#include <variant>
#include "utils/exception.h"
#include <fmt/format.h>
#include <rapidcheck.h>

namespace FlexFlow {

struct BinarySPDecompositionTree;

struct BinarySeriesSplit {
public:
  BinarySeriesSplit() = delete;
  explicit BinarySeriesSplit(BinarySPDecompositionTree const &,
                         BinarySPDecompositionTree const &);

  BinarySeriesSplit(BinarySeriesSplit const &) = default;

  bool operator==(BinarySeriesSplit const &) const;
  bool operator!=(BinarySeriesSplit const &) const;
  bool operator<(BinarySeriesSplit const &) const;

  BinarySPDecompositionTree const &left_child() const;
  BinarySPDecompositionTree const &right_child() const;

private:
  std::shared_ptr<BinarySPDecompositionTree> left_child_ptr;
  std::shared_ptr<BinarySPDecompositionTree> right_child_ptr;

private:
  std::tuple<
    BinarySPDecompositionTree const &,
    BinarySPDecompositionTree const &
  > tie() const;

  friend std::hash<BinarySeriesSplit>;
};

std::string format_as(BinarySeriesSplit const &);
std::ostream &operator<<(std::ostream &, BinarySeriesSplit const &);


struct BinaryParallelSplit {
public:
  BinaryParallelSplit() = delete;
  explicit BinaryParallelSplit(BinarySPDecompositionTree const &,
                         BinarySPDecompositionTree const &);

  BinaryParallelSplit(BinaryParallelSplit const &) = default;

  bool operator==(BinaryParallelSplit const &) const;
  bool operator!=(BinaryParallelSplit const &) const;
  bool operator<(BinaryParallelSplit const &) const;

  BinarySPDecompositionTree const &left_child() const;
  BinarySPDecompositionTree const &right_child() const;

private:
  std::shared_ptr<BinarySPDecompositionTree> left_child_ptr;
  std::shared_ptr<BinarySPDecompositionTree> right_child_ptr;

private:
  std::tuple<
    BinarySPDecompositionTree const &,
    BinarySPDecompositionTree const &
  > tie() const;

  friend std::hash<BinaryParallelSplit>;
};


std::string format_as(BinaryParallelSplit const &);
std::ostream &operator<<(std::ostream &, BinaryParallelSplit const &);

struct BinarySPDecompositionTree {
public:
  BinarySPDecompositionTree() = delete;
  explicit BinarySPDecompositionTree(BinarySeriesSplit const &);
  explicit BinarySPDecompositionTree(BinaryParallelSplit const &);
  explicit BinarySPDecompositionTree(Node const &);

  BinarySPDecompositionTree(BinarySPDecompositionTree const &) = default;

  bool operator==(BinarySPDecompositionTree const &) const;
  bool operator!=(BinarySPDecompositionTree const &) const;
  bool operator<(BinarySPDecompositionTree const &) const;

  SPDecompositionTreeNodeType get_node_type() const;

  BinarySeriesSplit const &require_series() const;
  BinaryParallelSplit const &require_parallel() const;
  Node const &require_node() const;

  template <typename T>
  bool has() const {
    return std::holds_alternative<T>(this->root);
  }

  template <typename T>
  T const &get() const {
    return std::get<T>(this->root);
  }

  template <typename Result, typename F>
  Result visit(F f) const {
    SPDecompositionTreeNodeType tree_node_type = this->get_node_type();
    switch (tree_node_type) {
      case SPDecompositionTreeNodeType::SERIES: {
        Result result = f(this->require_series());
        return result;
      }
      case SPDecompositionTreeNodeType::PARALLEL: {
        Result result = f(this->require_parallel());
        return result;
      }
      case SPDecompositionTreeNodeType::NODE: {
        Result result = f(this->require_node());
        return result;
      }
      default:
        throw mk_runtime_error(fmt::format("Unknown SPDecompositionTreeNodeType {}", tree_node_type));
    }
  }

  static BinarySPDecompositionTree parallel(BinarySPDecompositionTree const &lhs, BinarySPDecompositionTree const &rhs);
  static BinarySPDecompositionTree series(BinarySPDecompositionTree const &lhs, BinarySPDecompositionTree const &rhs);
  static BinarySPDecompositionTree node(Node const &n);
private:
  std::variant<BinarySeriesSplit, BinaryParallelSplit, Node> root;

private:
  std::tuple<decltype(root) const &> 
    tie() const;

  friend std::hash<BinarySPDecompositionTree>;
};

std::string format_as(BinarySPDecompositionTree const &);
std::ostream &operator<<(std::ostream &, BinarySPDecompositionTree const &);

} // namespace FlexFlow

namespace std {

template <>
struct hash<::FlexFlow::BinarySeriesSplit> {
  size_t operator()(::FlexFlow::BinarySeriesSplit const &) const;
};

template <>
struct hash<::FlexFlow::BinaryParallelSplit> {
  size_t operator()(::FlexFlow::BinaryParallelSplit const &) const;
};

template <>
struct hash<::FlexFlow::BinarySPDecompositionTree> {
  size_t operator()(::FlexFlow::BinarySPDecompositionTree const &) const;
};

} // namespace std

namespace nlohmann {

template <>
struct adl_serializer<::FlexFlow::BinarySeriesSplit> {
  static ::FlexFlow::BinarySeriesSplit from_json(json const &);
  static void to_json(json &, ::FlexFlow::BinarySeriesSplit const &);
};

template <>
struct adl_serializer<::FlexFlow::BinaryParallelSplit> {
  static ::FlexFlow::BinaryParallelSplit from_json(json const &);
  static void to_json(json &, ::FlexFlow::BinaryParallelSplit const &);
};

template <>
struct adl_serializer<::FlexFlow::BinarySPDecompositionTree> {
  static ::FlexFlow::BinarySPDecompositionTree from_json(json const &);
  static void to_json(json &, ::FlexFlow::BinarySPDecompositionTree const &);
};

} // namespace nlohmann

namespace rc {

template <>
struct Arbitrary<::FlexFlow::BinarySeriesSplit> {
  static Gen<::FlexFlow::BinarySeriesSplit> arbitrary();
};

template <>
struct Arbitrary<::FlexFlow::BinaryParallelSplit> {
  static Gen<::FlexFlow::BinaryParallelSplit> arbitrary();
};

template <>
struct Arbitrary<::FlexFlow::BinarySPDecompositionTree> {
  static Gen<::FlexFlow::BinarySPDecompositionTree> arbitrary();
};

} // namespace rc

#endif
