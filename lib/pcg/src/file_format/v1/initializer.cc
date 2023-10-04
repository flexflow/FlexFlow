#include "pcg/file_format/v1/initializer.h"

namespace FlexFlow {

V1GlorotInitializer to_v1(GlorotUniform const &i) {
  return {i.seed};
}

V1ZeroInitializer to_v1(ZeroInitializer const &i) {
  return {
      // No fields in ZeroInitializer.
  };
}

V1UniformInitializer to_v1(UniformInitializer const &i) {
  return {i.seed, i.min_val, i.max_val};
}

V1NormInitializer to_v1(NormInitializer const &i) {
  return {i.seed, i.mean, i.stddev};
}

V1ConstantInitializer to_v1(ConstantInitializer const &i) {
  return {to_v1(i.value)};
}

V1Initializer to_v1(Initializer const &i) {
  // There is surely a better way of doing this ...
  if (const auto* glorot = get_if<GlorotUniform>(&i))
    return to_v1(*glorot);
  else if (const auto* zero = get_if<ZeroInitializer>(&i))
    return to_v1(*zero);
  else if (const auto* uniform = get_if<UniformInitializer>(&i))
    return to_v1(*uniform);
  else if (const auto* norm = get_if<NormInitializer>(&i))
    return to_v1(*norm);
  else if (const auto* constant = get_if<ConstantInitializer>(&i))
    return to_v1(*constant);
  else
    NOT_REACHABLE();
}

} // namespace FlexFlow
