#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_VALUES_ARE_APPROXIMATELY_EQUAL_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_VALUES_ARE_APPROXIMATELY_EQUAL_H

#include "utils/overload.h"
#include <cmath>
#include <optional>

namespace FlexFlow {

bool floats_are_approximately_equal(
    float, float, std::optional<double> relative_error_allowed = std::nullopt);
bool doubles_are_approximately_equal(
    double,
    double,
    std::optional<double> relative_error_allowed = std::nullopt);

template <typename T>
bool values_are_approximately_equal(
    T l, T r, std::optional<double> relative_error_allowed = std::nullopt) {
  auto f = overload{
      [&](float ll, float rr) -> bool {
        return floats_are_approximately_equal(ll, rr, relative_error_allowed);
      },
      [&](double ll, double rr) -> bool {
        return doubles_are_approximately_equal(ll, rr, relative_error_allowed);
      },
      [&](int32_t ll, int32_t rr) -> bool { return ll == rr; },
      [&](int64_t ll, int64_t rr) -> bool { return ll == rr; },
      [&](bool ll, bool rr) -> bool { return ll == rr; },
  };

  return f(l, r);
}

} // namespace FlexFlow

#endif
