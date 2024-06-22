// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/op-attrs/include/op-attrs/ops/weight_attrs.struct.toml
/* proj-data
{
  "generated_from": "59f49374ffca95b2117b8940af1b6cac"
}
*/

#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_WEIGHT_ATTRS_DTG_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_WEIGHT_ATTRS_DTG_H

#include "fmt/format.h"
#include "nlohmann/json.hpp"
#include "rapidcheck.h"
#include <functional>
#include <ostream>
#include <tuple>

namespace FlexFlow {
struct WeightAttrs {
  bool operator==(WeightAttrs const &) const;
  bool operator!=(WeightAttrs const &) const;
  bool operator<(WeightAttrs const &) const;
  bool operator>(WeightAttrs const &) const;
  bool operator<=(WeightAttrs const &) const;
  bool operator>=(WeightAttrs const &) const;
};
} // namespace FlexFlow

namespace std {
template <>
struct hash<::FlexFlow::WeightAttrs> {
  size_t operator()(::FlexFlow::WeightAttrs const &) const;
};
} // namespace std

namespace nlohmann {
template <>
struct adl_serializer<::FlexFlow::WeightAttrs> {
  static ::FlexFlow::WeightAttrs from_json(json const &);
  static void to_json(json &, ::FlexFlow::WeightAttrs const &);
};
} // namespace nlohmann

namespace rc {
template <>
struct Arbitrary<::FlexFlow::WeightAttrs> {
  static Gen<::FlexFlow::WeightAttrs> arbitrary();
};
} // namespace rc

namespace FlexFlow {
std::string format_as(WeightAttrs const &);
std::ostream &operator<<(std::ostream &, WeightAttrs const &);
} // namespace FlexFlow

#endif // _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_WEIGHT_ATTRS_DTG_H
