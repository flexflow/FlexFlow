#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_FILE_FORMAT_V1_V1_BINARY_SP_DECOMPOSITION_JSON_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_FILE_FORMAT_V1_V1_BINARY_SP_DECOMPOSITION_JSON_H

#include "pcg/file_format/v1/v1_binary_sp_decomposition/v1_binary_sp_decomposition.dtg.h"
#include <nlohmann/json.hpp>

namespace nlohmann {

template <>
struct adl_serializer<::FlexFlow::V1BinarySPDecomposition> {
  static ::FlexFlow::V1BinarySPDecomposition from_json(json const &);
  static void to_json(json &, ::FlexFlow::V1BinarySPDecomposition const &);
};

template <>
struct adl_serializer<::FlexFlow::V1BinarySeriesSplit> {
  static ::FlexFlow::V1BinarySeriesSplit from_json(json const &);
  static void to_json(json &, ::FlexFlow::V1BinarySeriesSplit const &);
};

template <>
struct adl_serializer<::FlexFlow::V1BinaryParallelSplit> {
  static ::FlexFlow::V1BinaryParallelSplit from_json(json const &);
  static void to_json(json &, ::FlexFlow::V1BinaryParallelSplit const &);
};

} // namespace nlohmann

#endif
