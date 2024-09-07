#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_JSON_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_JSON_H

#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree.h"
#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_left_child.h"
#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_right_child.h"
#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/visit.h"
#include <nlohmann/json.hpp>
#include "utils/exception.h"

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
    j["left_child"] = get_left_child(v);
    j["right_child"] = get_right_child(v);
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
    j["left_child"] = get_left_child(v);
    j["right_child"] = get_right_child(v);
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
    ::FlexFlow::visit<std::monostate>(v, ::FlexFlow::overload{
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


#endif
