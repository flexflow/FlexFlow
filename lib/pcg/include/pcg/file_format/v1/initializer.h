#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_INITIALIZER_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_INITIALIZER_H

#include "utils/visitable.h"
#include "utils/json.h"
#include "utils/variant.h"
#include "pcg/file_format/keyed_variant.h"
#include "data_type.h"
#include "visit_struct/visit_struct_intrusive.hpp"
#include "utils/required.h"

namespace FlexFlow {

struct V1GlorotInitializer : public use_visitable_cmp<V1GlorotInitializer> {
  req<int> seed;
};
FF_VISITABLE_STRUCT(V1GlorotInitializer, seed);

struct V1ZeroInitializer : public use_visitable_cmp<V1ZeroInitializer> { };
FF_VISITABLE_STRUCT(V1ZeroInitializer);

struct V1UniformInitializer : public use_visitable_cmp<V1UniformInitializer> {
  req<int> seed;
  req<float> min_val;
  req<float> max_val;
};
FF_VISITABLE_STRUCT(V1UniformInitializer, seed, min_val, max_val);

struct V1NormInitializer : public use_visitable_cmp<V1NormInitializer> {
  req<int> seed;
  req<float> mean;
  req<float> stddev;
};
FF_VISITABLE_STRUCT(V1NormInitializer, seed, mean, stddev);

struct V1ConstantInitializer : public use_visitable_cmp<V1ConstantInitializer> {
  V1DataTypeValue value;
};
FF_VISITABLE_STRUCT(V1ConstantInitializer, value);

using V1InitializerVariant = variant<V1GlorotInitializer, V1ZeroInitializer, V1UniformInitializer, V1NormInitializer, V1ConstantInitializer>;

enum class V1InitializerType { 
  GLOROT = index_of_type<V1GlorotInitializer, V1InitializerVariant>::value,
  ZERO = index_of_type<V1ZeroInitializer, V1InitializerVariant>::value,
  UNIFORM = index_of_type<V1UniformInitializer, V1InitializerVariant>::value,
  NORMAL = index_of_type<V1NormInitializer, V1InitializerVariant>::value,
  CONSTANT = index_of_type<V1ConstantInitializer, V1InitializerVariant>::value,
};

NLOHMANN_JSON_SERIALIZE_ENUM(V1InitializerType,
                             {{V1InitializerType::GLOROT, "GLOROT"},
                              {V1InitializerType::ZERO, "ZERO"},
                              {V1InitializerType::UNIFORM, "UNIFORM"},
                              {V1InitializerType::NORMAL, "NORMAL"},
                              {V1InitializerType::CONSTANT, "CONSTANT"}});

using V1Initializer = KeyedVariant<V1InitializerType, V1InitializerVariant>;
}

namespace FlexFlow {
static_assert (is_jsonable<::FlexFlow::V1GlorotInitializer>::value, "");
static_assert (is_jsonable<::FlexFlow::V1ZeroInitializer>::value, "");
static_assert (is_jsonable<::FlexFlow::V1UniformInitializer>::value, "");
static_assert (is_jsonable<::FlexFlow::V1NormInitializer>::value, "");
static_assert (is_jsonable<::FlexFlow::V1ConstantInitializer>::value, "");
static_assert (is_jsonable<::FlexFlow::V1InitializerType>::value, "");
static_assert (is_json_serializable<::FlexFlow::V1Initializer>::value, "");
static_assert (is_json_deserializable<::FlexFlow::V1Initializer>::value, "");
static_assert (is_jsonable<::FlexFlow::V1Initializer>::value, "");
}

#endif
