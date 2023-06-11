#include "pcg/initializer.h"

namespace FlexFlow {

GlorotUniform::GlorotUniform(int _seed) : seed(_seed) {}

UniformInitializer::UniformInitializer(int _seed, float _min, float _max)
    : seed(_seed), min_val(_min), max_val(_max) {}

NormInitializer::NormInitializer(int _seed, float _mean, float _stddev)
    : seed(_seed), mean(_mean), stddev(_stddev) {}

ConstantInitializer::ConstantInitializer(DataTypeValue const &_value)
    : value(_value) {}

} // namespace FlexFlow
