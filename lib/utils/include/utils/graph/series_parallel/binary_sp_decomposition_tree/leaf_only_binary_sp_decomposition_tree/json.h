#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_LEAF_ONLY_BINARY_SP_DECOMPOSITION_TREE_JSON_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_LEAF_ONLY_BINARY_SP_DECOMPOSITION_TREE_JSON_H

// #include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree.dtg.h"
// #include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree_visitor.dtg.h"
// #include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/wrap.h"
// #include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/visit.h"
// #include "utils/json/check_is_json_deserializable.h"
// #include "utils/json/check_is_json_serializable.h"
// #include "utils/fmt/json.h"
// #include <nlohmann/json.hpp>

namespace nlohmann {

// template <typename LeafLabel>
// struct adl_serializer<::FlexFlow::LeafOnlyBinarySPDecompositionTree<LeafLabel>> {
//   static ::FlexFlow::LeafOnlyBinarySPDecompositionTree<LeafLabel> from_json(json const &j) {
//     CHECK_IS_JSON_SERIALIZABLE(LeafLabel); 
//
//     using namespace ::FlexFlow;
//
//     using Tree = LeafOnlyBinarySPDecompositionTree<LeafLabel>;
//
//     std::string type = j.at("type").get<std::string>();
//
//     if (type == "series") {
//       return leaf_only_binary_sp_tree_wrap_series_split(
//         LeafOnlyBinarySeriesSplit{
//           /*lhs=*/j.at("left_child").get<Tree>(),
//           /*rhs=*/j.at("right_child").get<Tree>(),
//         });
//     } else if (type == "parallel") {
//       return leaf_only_binary_sp_tree_wrap_parallel_split(
//         LeafOnlyBinaryParallelSplit{ 
//           /*lhs=*/j.at("left_child").get<Tree>(),
//           /*rhs=*/j.at("right_child").get<Tree>(),
//         });
//     } else if (type == "leaf") {
//       return leaf_only_binary_sp_tree_wrap_leaf(j.at("label").get<LeafLabel>()); 
//     } else {
//       throw mk_runtime_error(fmt::format("Unknown json type value for LeafOnlyBinarySPDecompositionTree \"{}\" in json object: {}", type, j));
//     }
//   }
//
//   static void to_json(json &j, ::FlexFlow::LeafOnlyBinarySPDecompositionTree<LeafLabel> const &tree) {
//     CHECK_IS_JSON_DESERIALIZABLE(LeafLabel); 
//
//     using namespace FlexFlow;
//
//     using Tree = LeafOnlyBinarySPDecompositionTree<LeafLabel>;
//
//     auto visitor = LeafOnlyBinarySPDecompositionTreeVisitor<std::monostate, LeafLabel>{
//       /*series_func=*/[&](LeafOnlyBinarySeriesSplit<LeafLabel> const &split) {
//         j["type"] = "series"; 
//         j["left_child"] = split.lhs;
//         j["right_child"] = split.rhs;
//         return std::monostate{};
//       },
//       /*parallel_func=*/[&](LeafOnlyBinaryParallelSplit<LeafLabel> const &split) {
//         j["type"] = "parallel"; 
//         j["left_child"] = split.lhs;
//         j["right_child"] = split.rhs;
//         return std::monostate{};
//       },
//       /*leaf_func=*/[&](LeafLabel const &leaf_label) {
//         j["type"] = "leaf"; 
//         j["label"] = leaf_label;
//         return std::monostate{};
//       },
//     };
//
//     visit(tree, visitor);
//   }
// };

} // namespace nlohmann

#endif
