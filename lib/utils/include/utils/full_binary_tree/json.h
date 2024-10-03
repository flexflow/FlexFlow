#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_JSON_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_JSON_H

#include "utils/exception.h"
#include "utils/full_binary_tree/full_binary_tree.dtg.h"
#include "utils/full_binary_tree/get_left_child.h"
#include "utils/full_binary_tree/get_right_child.h"
#include "utils/full_binary_tree/visit.h"
#include "utils/overload.h"
#include <nlohmann/json.hpp>

namespace nlohmann {

template <typename ParentLabel, typename LeafLabel>
struct adl_serializer<::FlexFlow::FullBinaryTreeParentNode<ParentLabel, LeafLabel>> {
  static ::FlexFlow::FullBinaryTreeParentNode<ParentLabel, LeafLabel> from_json(json const &j) {
    return ::FlexFlow::FullBinaryTreeParentNode<ParentLabel, LeafLabel>{
        j.at("left_child")
            .template get<::FlexFlow::FullBinaryTreeParentNode<ParentLabel, LeafLabel>>(),
        j.at("right_child")
            .template get<::FlexFlow::FullBinaryTreeParentNode<ParentLabel, LeafLabel>>(),
    };
  }

  static void to_json(json &j,
                      ::FlexFlow::FullBinaryTreeParentNode<ParentLabel, LeafLabel> const &v) {
    j["__type"] = "FullBinaryTreeParentNode";
    j["left_child"] = get_left_child(v);
    j["right_child"] = get_right_child(v);
  }
};

template <typename ParentLabel, typename LeafLabel>
struct adl_serializer<::FlexFlow::FullBinaryTree<ParentLabel, LeafLabel>> {
  static ::FlexFlow::FullBinaryTree<ParentLabel, LeafLabel> from_json(json const &j) {
    std::string key = j.at("type").get<std::string>();

    if (key == "parent") {
      return ::FlexFlow::FullBinaryTree<ParentLabel, LeafLabel>{
          j.at("value").get<::FlexFlow::FullBinaryTreeParentNode<ParentLabel, LeafLabel>>(),
      };
    } else if (key == "leaf") {
      return ::FlexFlow::FullBinaryTree<ParentLabel, LeafLabel>{
          j.at("value").get<LeafLabel>(),
      };
    } else {
      throw ::FlexFlow::mk_runtime_error(
          fmt::format("Unknown json type key: {}", key));
    }
  }

  static void
      to_json(json &j,
              ::FlexFlow::FullBinaryTree<ParentLabel, LeafLabel> const &v) {
    j["__type"] = "FullBinaryTree";
    ::FlexFlow::visit<std::monostate>(
        v,
        ::FlexFlow::overload{
            [&](::FlexFlow::FullBinaryTreeParentNode<ParentLabel, LeafLabel> const &s) {
              j["type"] = "parent";
              j["value"] = s;
              return std::monostate{};
            },
            [&](LeafLabel const &t) {
              j["type"] = "leaf";
              j["value"] = t;
              return std::monostate{};
            },
        });
  }
};

} // namespace nlohmann

#endif
