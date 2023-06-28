#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_DATA_TYPE_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_DATA_TYPE_H

#include "utils/variant.h"
#include "utils/fp16.h"
#include "pcg/file_format/keyed_variant.h"
#include "utils/json.h"

namespace FlexFlow {

using V1DataTypeValueVariant = variant<bool, int32_t, int64_t, half, float, double>;

enum class V1DataType { 
  BOOL = index_of_type<bool, V1DataTypeValueVariant>::value, 
  INT32 = index_of_type<int32_t, V1DataTypeValueVariant>::value,
  INT64 = index_of_type<int64_t, V1DataTypeValueVariant>::value,
  HALF = index_of_type<half, V1DataTypeValueVariant>::value,
  FLOAT = index_of_type<float, V1DataTypeValueVariant>::value,
  DOUBLE = index_of_type<double, V1DataTypeValueVariant>::value
};

NLOHMANN_JSON_SERIALIZE_ENUM(V1DataType,
                             {{V1DataType::BOOL, "BOOL"},
                              {V1DataType::INT32, "INT32"},
                              {V1DataType::INT64, "INT64"},
                              {V1DataType::HALF, "HALF"},
                              {V1DataType::FLOAT, "FLOAT"},
                              {V1DataType::DOUBLE, "DOUBLE"}});

using V1DataTypeValue = KeyedVariant<V1DataType, V1DataTypeValueVariant>;

}

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

}

namespace FlexFlow {

void thing() {
  json j;
  auto g = j.get<V1DataTypeValue>();
}

/* static_assert(elements_satisfy<is_json_serializable, V1DataTypeValueVariant>::value, ""); */
/* static_assert(elements_satisfy<is_json_deserializable, V1DataTypeValueVariant>::value, ""); */
/* static_assert(is_jsonable<half>::value, ""); */
/* static_assert(is_json_serializable<V1DataTypeValue>::value, ""); */
/* static_assert(is_json_deserializable<V1DataTypeValue>::value, ""); */
/* static_assert(is_jsonable<V1DataTypeValue>::value, ""); */
}

#endif
