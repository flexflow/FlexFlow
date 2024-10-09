#include "pcg/file_format/v1/v1_binary_sp_decomposition/json.h"
#include "utils/exception.h"
#include "utils/fmt/json.h"
#include "utils/overload.h"

using namespace ::FlexFlow;

namespace nlohmann {

V1BinarySPDecomposition
    adl_serializer<V1BinarySPDecomposition>::from_json(json const &j) {
  std::string type = j.at("type").get<std::string>();

  if (type == "series") {
    return V1BinarySPDecomposition{
        j.get<V1BinarySeriesSplit>(),
    };
  } else if (type == "parallel") {
    return V1BinarySPDecomposition{
        j.get<V1BinaryParallelSplit>(),
    };
  } else if (type == "leaf") {
    return V1BinarySPDecomposition{
        j.at("value").get<int>(),
    };
  } else {
    throw mk_runtime_error(fmt::format(
        "Unknown json type value for LeafOnlyBinarySPDecompositionTree \"{}\" "
        "in json object: {}",
        type,
        j));
  }
}

void adl_serializer<V1BinarySPDecomposition>::to_json(
    json &j, V1BinarySPDecomposition const &tree) {
  tree.visit<std::monostate>(overload{
      [&](V1BinarySeriesSplit const &split) {
        j = split;
        j["type"] = "series";
        return std::monostate{};
      },
      [&](V1BinaryParallelSplit const &split) {
        j = split;
        j["type"] = "parallel";
        return std::monostate{};
      },
      [&](int leaf) {
        j["value"] = leaf;
        j["type"] = "leaf";
        return std::monostate{};
      },
  });
}

V1BinarySeriesSplit
    adl_serializer<V1BinarySeriesSplit>::from_json(json const &j) {
  return V1BinarySeriesSplit{
      /*lhs=*/j.at("left_child").get<V1BinarySPDecomposition>(),
      /*rhs=*/j.at("right_child").get<V1BinarySPDecomposition>(),
  };
}

void adl_serializer<V1BinarySeriesSplit>::to_json(
    json &j, V1BinarySeriesSplit const &series) {
  j["left_child"] = series.get_left_child();
  j["right_child"] = series.get_right_child();
}

V1BinaryParallelSplit
    adl_serializer<V1BinaryParallelSplit>::from_json(json const &j) {
  return V1BinaryParallelSplit{
      /*lhs=*/j.at("left_child").get<V1BinarySPDecomposition>(),
      /*rhs=*/j.at("right_child").get<V1BinarySPDecomposition>(),
  };
}

void adl_serializer<V1BinaryParallelSplit>::to_json(
    json &j, V1BinaryParallelSplit const &series) {
  j["left_child"] = series.get_left_child();
  j["right_child"] = series.get_right_child();
}

} // namespace nlohmann
