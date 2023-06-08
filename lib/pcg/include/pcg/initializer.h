#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_INITIALIZER_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_INITIALIZER_H

#include "op-attrs/datatype.h"
#include "utils/visitable.h"

namespace FlexFlow {

class GlorotUniform : public use_visitable_cmp<GlorotUniform> {
public:
  GlorotUniform() = delete;
  GlorotUniform(int seed);

public:
  int seed;
  /* float scale; */
  /* DataType data_type; */
};

class ZeroInitializer : public use_visitable_cmp<ZeroInitializer> {
public:
  ZeroInitializer() = default;
};

class UniformInitializer : public use_visitable_cmp<UniformInitializer> {
public:
  UniformInitializer(int seed, float min, float max);

public:
  int seed;
  float min_val, max_val;
};

class NormInitializer : public use_visitable_cmp<NormInitializer> {
public:
  NormInitializer(int seed, float mean, float stddev);

public:
  int seed;
  float mean, stddev;
};

class ConstantInitializer : public use_visitable_cmp<ConstantInitializer> {
public:
  ConstantInitializer(DataTypeValue const &value);

public:
  DataTypeValue value;
};

using Initializer = variant<GlorotUniform, ZeroInitializer, UniformInitializer,
                            NormInitializer, ConstantInitializer>;

} // namespace FlexFlow

#endif
