// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/pcg/include/pcg/optimizers/sgd_optimizer_attrs.struct.toml
/* proj-data
{
  "generated_from": "d18c91cdddc760f1fb3990d2c817ee87"
}
*/

#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_OPTIMIZERS_SGD_OPTIMIZER_ATTRS_DTG_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_OPTIMIZERS_SGD_OPTIMIZER_ATTRS_DTG_H

#include "fmt/format.h"
#include "nlohmann/json.hpp"
#include "rapidcheck.h"
#include <functional>
#include <ostream>
#include <tuple>

namespace FlexFlow {
struct SGDOptimizerAttrs {
  SGDOptimizerAttrs() = delete;
  explicit SGDOptimizerAttrs(double const &lr,
                             double const &momentum,
                             bool const &nesterov,
                             double const &weight_decay);

  bool operator==(SGDOptimizerAttrs const &) const;
  bool operator!=(SGDOptimizerAttrs const &) const;
  bool operator<(SGDOptimizerAttrs const &) const;
  bool operator>(SGDOptimizerAttrs const &) const;
  bool operator<=(SGDOptimizerAttrs const &) const;
  bool operator>=(SGDOptimizerAttrs const &) const;
  double lr;
  double momentum;
  bool nesterov;
  double weight_decay;
};
} // namespace FlexFlow

namespace std {
template <>
struct hash<::FlexFlow::SGDOptimizerAttrs> {
  size_t operator()(::FlexFlow::SGDOptimizerAttrs const &) const;
};
} // namespace std

namespace nlohmann {
template <>
struct adl_serializer<::FlexFlow::SGDOptimizerAttrs> {
  static ::FlexFlow::SGDOptimizerAttrs from_json(json const &);
  static void to_json(json &, ::FlexFlow::SGDOptimizerAttrs const &);
};
} // namespace nlohmann

namespace rc {
template <>
struct Arbitrary<::FlexFlow::SGDOptimizerAttrs> {
  static Gen<::FlexFlow::SGDOptimizerAttrs> arbitrary();
};
} // namespace rc

namespace FlexFlow {
std::string format_as(SGDOptimizerAttrs const &);
std::ostream &operator<<(std::ostream &, SGDOptimizerAttrs const &);
} // namespace FlexFlow

#endif // _FLEXFLOW_LIB_PCG_INCLUDE_PCG_OPTIMIZERS_SGD_OPTIMIZER_ATTRS_DTG_H