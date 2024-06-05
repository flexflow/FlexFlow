// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/op-attrs/include/op-attrs/ops/replicate_attrs.struct.toml
/* proj-data
{
  "generated_from": "6d3ad4d10c24dae819ffee4592a72499"
}
*/

#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_REPLICATE_ATTRS_DTG_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_REPLICATE_ATTRS_DTG_H

#include "fmt/format.h"
#include "nlohmann/json.hpp"
#include "rapidcheck.h"
#include <functional>
#include <ostream>
#include <tuple>

namespace FlexFlow {
struct ReplicateAttrs {
  ReplicateAttrs() = delete;
  ReplicateAttrs(int const &replicate_degree);

  bool operator==(ReplicateAttrs const &) const;
  bool operator!=(ReplicateAttrs const &) const;
  bool operator<(ReplicateAttrs const &) const;
  bool operator>(ReplicateAttrs const &) const;
  bool operator<=(ReplicateAttrs const &) const;
  bool operator>=(ReplicateAttrs const &) const;
  int replicate_degree;
};
} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::ReplicateAttrs> {
  size_t operator()(FlexFlow::ReplicateAttrs const &) const;
};
} // namespace std

namespace nlohmann {
template <>
struct adl_serializer<FlexFlow::ReplicateAttrs> {
  static FlexFlow::ReplicateAttrs from_json(json const &);
  static void to_json(json &, FlexFlow::ReplicateAttrs const &);
};
} // namespace nlohmann

namespace rc {
template <>
struct Arbitrary<FlexFlow::ReplicateAttrs> {
  static Gen<FlexFlow::ReplicateAttrs> arbitrary();
};
} // namespace rc

namespace FlexFlow {
std::string format_as(ReplicateAttrs const &);
std::ostream &operator<<(std::ostream &, ReplicateAttrs const &);
} // namespace FlexFlow

#endif // _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_REPLICATE_ATTRS_DTG_H
