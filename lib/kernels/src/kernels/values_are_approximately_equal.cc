#include "kernels/values_are_approximately_equal.h"

namespace FlexFlow {

template bool
    values_are_approximately_equal(float, float, std::optional<double>);
template bool
    values_are_approximately_equal(double, double, std::optional<double>);
template bool
    values_are_approximately_equal(int32_t, int32_t, std::optional<double>);
template bool
    values_are_approximately_equal(int64_t, int64_t, std::optional<double>);
template bool values_are_approximately_equal(bool, bool, std::optional<double>);

bool floats_are_approximately_equal(
    float l, float r, std::optional<double> relative_error_allowed) {
  float max_magnitude = std::max(std::fabs(l), std::fabs(r));
  return std::fabs(l - r) <
         relative_error_allowed.value_or(0.00001) * max_magnitude;
}

bool doubles_are_approximately_equal(
    double l, double r, std::optional<double> relative_error_allowed) {
  float max_magnitude = std::max(std::fabs(l), std::fabs(r));
  return std::fabs(l - r) <
         relative_error_allowed.value_or(0.00001) * max_magnitude;
}

} // namespace FlexFlow
