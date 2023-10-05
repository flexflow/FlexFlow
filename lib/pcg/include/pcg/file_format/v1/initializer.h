#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_INITIALIZER_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_INITIALIZER_H

#include "datatype.h"
#include "pcg/initializer.h"
#include "utils/json.h"
#include "utils/required.h"
#include "utils/variant.h"
#include "utils/visitable.h"
#include "visit_struct/visit_struct_intrusive.hpp"

namespace FlexFlow {

struct V1GlorotInitializer {
  req<int> seed;
};
FF_VISITABLE_STRUCT(V1GlorotInitializer, seed);
CHECK_IS_JSONABLE(V1GlorotInitializer);

V1GlorotInitializer to_v1(GlorotUniform const &i);
GlorotUniform from_v1(V1GlorotInitializer const &vi);

struct V1ZeroInitializer {};
FF_VISITABLE_STRUCT(V1ZeroInitializer);

V1ZeroInitializer to_v1(ZeroInitializer const &i);
ZeroInitializer from_v1(V1ZeroInitializer const &vi);

struct V1UniformInitializer {
  req<int> seed;
  req<float> min_val;
  req<float> max_val;
};
FF_VISITABLE_STRUCT(V1UniformInitializer, seed, min_val, max_val);
CHECK_IS_JSONABLE(V1UniformInitializer);

V1UniformInitializer to_v1(UniformInitializer const &i);
UniformInitializer from_v1(V1UniformInitializer const &vi);

struct V1NormInitializer {
  req<int> seed;
  req<float> mean;
  req<float> stddev;
};
FF_VISITABLE_STRUCT(V1NormInitializer, seed, mean, stddev);
CHECK_IS_JSONABLE(V1NormInitializer);

V1NormInitializer to_v1(NormInitializer const &i);
NormInitializer from_v1(V1NormInitializer const &vi);

struct V1ConstantInitializer {
  req<V1DataTypeValue> value;
};
FF_VISITABLE_STRUCT(V1ConstantInitializer, value);
CHECK_IS_JSONABLE(V1ConstantInitializer);

V1ConstantInitializer to_v1(ConstantInitializer const &i);
ConstantInitializer from_v1(V1ConstantInitializer const &vi);

using V1Initializer = variant<V1GlorotInitializer,
                              V1ZeroInitializer,
                              V1UniformInitializer,
                              V1NormInitializer,
                              V1ConstantInitializer>;

V1Initializer to_v1(Initializer const &i);
Initializer from_v1(V1Initializer const &vi);

} // namespace FlexFlow

#endif
