#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_DATA_TYPE_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_DATA_TYPE_H

#include "utils/fp16.h"
#include "utils/json.h"
#include "utils/variant.h"

namespace FlexFlow {

using V1DataTypeValue = variant<bool, int32_t, int64_t, half, float, double>;

enum class V1DataType {
  BOOL,
  INT32,
  INT64,
  HALF,
  FLOAT,
  DOUBLE,
};

NLOHMANN_JSON_SERIALIZE_ENUM(V1DataType,
                             {{V1DataType::BOOL, "BOOL"},
                              {V1DataType::INT32, "INT32"},
                              {V1DataType::INT64, "INT64"},
                              {V1DataType::HALF, "HALF"},
                              {V1DataType::FLOAT, "FLOAT"},
                              {V1DataType::DOUBLE, "DOUBLE"}});

} // namespace FlexFlow

namespace nlohmann {

template <>
struct adl_serializer<half> {
  static void to_json(json &j, half h) {
    j = json{static_cast<float>(h)};
  }

  static void from_json(json const &j, half &h) {
    h = static_cast<half>(j.get<float>());
  }
};

} // namespace nlohmann

namespace FlexFlow {
static_assert(is_jsonable<half>::value, "");
static_assert(is_json_serializable<V1DataTypeValue>::value, "");
static_assert(is_json_deserializable<V1DataTypeValue>::value, "");
static_assert(is_jsonable<V1DataTypeValue>::value, "");
} // namespace FlexFlow

#endif
