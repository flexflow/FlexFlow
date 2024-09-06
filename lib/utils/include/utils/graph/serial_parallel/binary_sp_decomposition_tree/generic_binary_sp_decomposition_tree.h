#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_H

#include "utils/exception.h"
#include "utils/graph/node/node.dtg.h"
#include "utils/graph/serial_parallel/sp_decomposition_tree_node_type.dtg.h"
#include "utils/hash-utils.h"
#include "utils/hash/tuple.h"
#include "utils/overload.h"
#include <fmt/format.h>
#include <rapidcheck.h>
#include <variant>

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

  GenericBinarySPDecompositionTree<T> const &left_child() const {
    return *this->left_child_ptr;
  }
  GenericBinarySPDecompositionTree<T> const &right_child() const {
    return *this->right_child_ptr;
  }

private:
  std::shared_ptr<GenericBinarySPDecompositionTree<T>> left_child_ptr;
  std::shared_ptr<GenericBinarySPDecompositionTree<T>> right_child_ptr;

private:
  std::tuple<GenericBinarySPDecompositionTree<T> const &,
             GenericBinarySPDecompositionTree<T> const &>
      tie() const {
    return std::tie(this->left_child(), this->right_child());
  }

  friend std::hash<GenericBinarySeriesSplit>;
};

template <typename T>
std::string format_as(GenericBinarySeriesSplit<T> const &s) {
  return fmt::format(
      "<BinarySeriesSplit {} {}>", s.left_child(), s.right_child());
}

template <typename T>
std::ostream &operator<<(std::ostream &s,
                         GenericBinarySeriesSplit<T> const &x) {
  return (s << fmt::to_string(x));
}

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

  GenericBinarySPDecompositionTree<T> const &left_child() const {
    return *this->left_child_ptr;
  }

  GenericBinarySPDecompositionTree<T> const &right_child() const {
    return *this->right_child_ptr;
  }

private:
  std::shared_ptr<GenericBinarySPDecompositionTree<T>> left_child_ptr;
  std::shared_ptr<GenericBinarySPDecompositionTree<T>> right_child_ptr;

private:
  std::tuple<GenericBinarySPDecompositionTree<T> const &,
             GenericBinarySPDecompositionTree<T> const &>
      tie() const {
    return std::tie(this->left_child(), this->right_child());
  }

  friend std::hash<GenericBinaryParallelSplit>;
};

template <typename T>
std::string format_as(GenericBinaryParallelSplit<T> const &s) {
  return fmt::format(
      "<BinaryParallelSplit {} {}>", s.left_child(), s.right_child());
}

template <typename T>
std::ostream &operator<<(std::ostream &s,
                         GenericBinaryParallelSplit<T> const &x) {
  return (s << fmt::to_string(x));
}

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

  SPDecompositionTreeNodeType get_node_type() const {
    // implemented using std::visit as opposed to this->visit because
    // this->visit is implemented using this function, so doing so would
    // create infinite recursion
    return std::visit(
        overload{
            [](GenericBinarySeriesSplit<T> const &) {
              return SPDecompositionTreeNodeType::SERIES;
            },
            [](GenericBinaryParallelSplit<T> const &) {
              return SPDecompositionTreeNodeType::PARALLEL;
            },
            [](T const &) { return SPDecompositionTreeNodeType::NODE; },
        },
        this->root);
  }

  GenericBinarySeriesSplit<T> const &require_series() const {
    return this->get<GenericBinarySeriesSplit<T>>();
  }

  GenericBinaryParallelSplit<T> const &require_parallel() const {
    return this->get<GenericBinaryParallelSplit<T>>();
  }

  T const &require_node() const {
    return this->get<T>();
  }

  template <typename TT>
  bool has() const {
    return std::holds_alternative<TT>(this->root);
  }

  template <typename TT>
  TT const &get() const {
    return std::get<TT>(this->root);
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
        throw mk_runtime_error(fmt::format(
            "Unknown GenericSPDecompositionTreeNodeType {}", tree_node_type));
    }
  }

  static GenericBinarySPDecompositionTree
      parallel(GenericBinarySPDecompositionTree const &lhs,
               GenericBinarySPDecompositionTree const &rhs);
  static GenericBinarySPDecompositionTree
      series(GenericBinarySPDecompositionTree const &lhs,
             GenericBinarySPDecompositionTree const &rhs);
  static GenericBinarySPDecompositionTree node(T const &n);

private:
  std::variant<GenericBinarySeriesSplit<T>, GenericBinaryParallelSplit<T>, T>
      root;

private:
  std::tuple<decltype(root) const &> tie() const {
    return std::tie(this->root);
  }

  friend std::hash<GenericBinarySPDecompositionTree>;
};

template <typename T>
std::string format_as(GenericBinarySPDecompositionTree<T> const &t) {
  return t.template visit<std::string>(overload{
      [](GenericBinarySeriesSplit<T> const &s) {
        return fmt::format("<GenericBinarySPDecompositionTree {}>", s);
      },
      [](GenericBinaryParallelSplit<T> const &s) {
        return fmt::format("<GenericBinarySPDecompositionTree {}>", s);
      },
      [](T const &t) {
        return fmt::format("<BinarySPDecompositionTree {}>", t);
      },
  });
}

template <typename T>
std::ostream &operator<<(std::ostream &s,
                         GenericBinarySPDecompositionTree<T> const &t) {
  return (s << fmt::to_string(t));
}

template <typename T, typename F, typename TT = std::invoke_result_t<F, T>>
GenericBinarySPDecompositionTree<TT>
    transform(GenericBinarySPDecompositionTree<T> const &tt, F f) {
  return tt.template visit<GenericBinarySPDecompositionTree<TT>>(overload{
      [&](GenericBinarySeriesSplit<T> const &s) {
        return GenericBinarySPDecompositionTree<TT>{
            GenericBinarySeriesSplit<TT>{
                transform(s.left_child(), f),
                transform(s.right_child(), f),
            },
        };
      },
      [&](GenericBinaryParallelSplit<T> const &s) {
        return GenericBinarySPDecompositionTree<TT>{
            GenericBinaryParallelSplit<TT>{
                transform(s.left_child(), f),
                transform(s.right_child(), f),
            },
        };
      },
      [&](T const &t) {
        return GenericBinarySPDecompositionTree<TT>{
            f(t),
        };
      },
  });
}

} // namespace FlexFlow

namespace std {

template <typename T>
struct hash<::FlexFlow::GenericBinarySeriesSplit<T>> {
  size_t operator()(::FlexFlow::GenericBinarySeriesSplit<T> const &s) const {
    return get_std_hash(s.tie());
  }
};

template <typename T>
struct hash<::FlexFlow::GenericBinaryParallelSplit<T>> {
  size_t operator()(::FlexFlow::GenericBinaryParallelSplit<T> const &s) const {
    return get_std_hash(s.tie());
  }
};

template <typename T>
struct hash<::FlexFlow::GenericBinarySPDecompositionTree<T>> {
  size_t operator()(
      ::FlexFlow::GenericBinarySPDecompositionTree<T> const &s) const {
    return get_std_hash(s.tie());
  }
};

} // namespace std

namespace nlohmann {

template <typename T>
struct adl_serializer<::FlexFlow::GenericBinarySeriesSplit<T>> {
  static ::FlexFlow::GenericBinarySeriesSplit<T> from_json(json const &j) {
    return ::FlexFlow::GenericBinarySeriesSplit<T>{
        j.at("left_child")
            .template get<::FlexFlow::GenericBinarySPDecompositionTree<T>>(),
        j.at("right_child")
            .template get<::FlexFlow::GenericBinarySPDecompositionTree<T>>(),
    };
  }

  static void to_json(json &j,
                      ::FlexFlow::GenericBinarySeriesSplit<T> const &v) {
    j["__type"] = "GenericBinarySeriesSplit";
    j["left_child"] = v.left_child();
    j["right_child"] = v.right_child();
  }
};

template <typename T>
struct adl_serializer<::FlexFlow::GenericBinaryParallelSplit<T>> {
  static ::FlexFlow::GenericBinaryParallelSplit<T> from_json(json const &j) {
    return ::FlexFlow::GenericBinaryParallelSplit<T>{
        j.at("left_child")
            .template get<::FlexFlow::GenericBinarySPDecompositionTree<T>>(),
        j.at("right_child")
            .template get<::FlexFlow::GenericBinarySPDecompositionTree<T>>(),
    };
  }

  static void to_json(json &j,
                      ::FlexFlow::GenericBinaryParallelSplit<T> const &v) {
    j["__type"] = "GenericBinaryParallelSplit";
    j["left_child"] = v.left_child();
    j["right_child"] = v.right_child();
  }
};

template <typename T>
struct adl_serializer<::FlexFlow::GenericBinarySPDecompositionTree<T>> {
  static ::FlexFlow::GenericBinarySPDecompositionTree<T>
      from_json(json const &j) {
    std::string key = j.at("type").get<std::string>();

    if (key == "series") {
      return ::FlexFlow::GenericBinarySPDecompositionTree<T>{
          j.at("value").get<::FlexFlow::GenericBinarySeriesSplit<T>>(),
      };
    } else if (key == "parallel") {
      return ::FlexFlow::GenericBinarySPDecompositionTree<T>{
          j.at("value").get<::FlexFlow::GenericBinaryParallelSplit<T>>(),
      };
    } else if (key == "node") {
      return ::FlexFlow::GenericBinarySPDecompositionTree<T>{
          j.at("value").get<T>(),
      };
    } else {
      throw ::FlexFlow::mk_runtime_error(
          fmt::format("Unknown json type key: {}", key));
    }
  }

  static void
      to_json(json &j,
              ::FlexFlow::GenericBinarySPDecompositionTree<T> const &v) {
    j["__type"] = "BinarySPDecompositionTree";
    v.template visit<std::monostate>(::FlexFlow::overload{
        [&](::FlexFlow::GenericBinarySeriesSplit<T> const &s) {
          j["type"] = "series";
          j["value"] = s;
          return std::monostate{};
        },
        [&](::FlexFlow::GenericBinaryParallelSplit<T> const &p) {
          j["type"] = "parallel";
          j["value"] = p;
          return std::monostate{};
        },
        [&](T const &t) {
          j["type"] = "node";
          j["value"] = t;
          return std::monostate{};
        },
    });
  }
};

} // namespace nlohmann

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
