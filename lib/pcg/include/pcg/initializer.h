#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_INITIALIZER_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_INITIALIZER_H

#include "op-attrs/datatype.h"
#include "utils/required.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct GlorotUniform {
  req<int> seed;
  /* float scale; */
  /* DataType data_type; */
};
FF_VISITABLE_STRUCT(GlorotUniform, seed);

struct ZeroInitializer {
  ZeroInitializer() = default;
};
FF_VISITABLE_STRUCT(ZeroInitializer);

struct UniformInitializer {
  req<int> seed;
  req<float> min_val;
  req<float> max_val;
};
FF_VISITABLE_STRUCT(UniformInitializer, seed, min_val, max_val);

struct NormInitializer {
  req<int> seed;
  req<float> mean;
  req<float> stddev;
};
FF_VISITABLE_STRUCT(NormInitializer, seed, mean, stddev);

struct ConstantInitializer {
  req<DataTypeValue> value;
};
FF_VISITABLE_STRUCT(ConstantInitializer, value);

using Initializer = variant<GlorotUniform,
                            ZeroInitializer,
                            UniformInitializer,
                            NormInitializer,
                            ConstantInitializer>;
CHECK_WELL_BEHAVED_VALUE_TYPE(Initializer);

} // namespace FlexFlow

#endif
