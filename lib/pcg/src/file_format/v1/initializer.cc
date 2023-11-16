#include "pcg/file_format/v1/initializer.h"

namespace FlexFlow {

V1GlorotInitializer to_v1(GlorotUniform const &i) {
  return {i.seed};
}

GlorotUniform from_v1(V1GlorotInitializer const &vi) {
  return {vi.seed};
}

V1ZeroInitializer to_v1(ZeroInitializer const &i) {
  return {
      // No fields in ZeroInitializer.
  };
}

ZeroInitializer from_v1(V1ZeroInitializer const &vi) {
  return {
      // No fields in V1ZeroInitializer
  };
}

V1UniformInitializer to_v1(UniformInitializer const &i) {
  return {i.seed, i.min_val, i.max_val};
}

UniformInitializer from_v1(V1UniformInitializer const &vi) {
  return {vi.seed, vi.min_val, vi.max_val};
}

V1NormInitializer to_v1(NormInitializer const &i) {
  return {i.seed, i.mean, i.stddev};
}

NormInitializer from_v1(V1NormInitializer const &vi) {
  return {vi.seed, vi.mean, vi.stddev};
}

V1ConstantInitializer to_v1(ConstantInitializer const &i) {
  return {to_v1(i.value)};
}

ConstantInitializer from_v1(V1ConstantInitializer const &vi) {
  return {from_v1(vi.value)};
}

V1Initializer to_v1(Initializer const &i) {
  // There is surely a better way of doing this ...
  if (auto const *glorot = get_if<GlorotUniform>(&i)) {
    return to_v1(*glorot);
  } else if (auto const *zero = get_if<ZeroInitializer>(&i)) {
    return to_v1(*zero);
  } else if (auto const *uniform = get_if<UniformInitializer>(&i)) {
    return to_v1(*uniform);
  } else if (auto const *norm = get_if<NormInitializer>(&i)) {
    return to_v1(*norm);
  } else if (auto const *constant = get_if<ConstantInitializer>(&i)) {
    return to_v1(*constant);
  } else {
    NOT_REACHABLE();
  }
}

Initializer from_v1(V1Initializer const &vi) {
  // There is surely a better way of doing this ...
  if (auto const *glorot = get_if<V1GlorotInitializer>(&vi)) {
    return from_v1(*glorot);
  } else if (auto const *zero = get_if<V1ZeroInitializer>(&vi)) {
    return from_v1(*zero);
  } else if (auto const *uniform = get_if<V1UniformInitializer>(&vi)) {
    return from_v1(*uniform);
  } else if (auto const *norm = get_if<V1NormInitializer>(&vi)) {
    return from_v1(*norm);
  } else if (auto const *constant = get_if<V1ConstantInitializer>(&vi)) {
    return from_v1(*constant);
  } else {
    NOT_REACHABLE();
  }
}

} // namespace FlexFlow
