// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/op-attrs/include/op-attrs/ops/cast_attrs.struct.toml
/* proj-data
{
  "generated_from": "902985a57f18e36925e35d90701329fa"
}
*/

#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_CAST_ATTRS_DTG_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_CAST_ATTRS_DTG_H

#include "fmt/format.h"
#include "nlohmann/json.hpp"
#include "op-attrs/datatype.dtg.h"
#include "rapidcheck.h"
#include <functional>
#include <ostream>
#include <tuple>

namespace FlexFlow {
struct CastAttrs {
  CastAttrs() = delete;
  explicit CastAttrs(::FlexFlow::DataType const &dtype);

  bool operator==(CastAttrs const &) const;
  bool operator!=(CastAttrs const &) const;
  bool operator<(CastAttrs const &) const;
  bool operator>(CastAttrs const &) const;
  bool operator<=(CastAttrs const &) const;
  bool operator>=(CastAttrs const &) const;
  ::FlexFlow::DataType dtype;
};
} // namespace FlexFlow

namespace std {
template <>
struct hash<::FlexFlow::CastAttrs> {
  size_t operator()(::FlexFlow::CastAttrs const &) const;
};
} // namespace std

namespace nlohmann {
template <>
struct adl_serializer<::FlexFlow::CastAttrs> {
  static ::FlexFlow::CastAttrs from_json(json const &);
  static void to_json(json &, ::FlexFlow::CastAttrs const &);
};
} // namespace nlohmann

namespace rc {
template <>
struct Arbitrary<::FlexFlow::CastAttrs> {
  static Gen<::FlexFlow::CastAttrs> arbitrary();
};
} // namespace rc

namespace FlexFlow {
std::string format_as(CastAttrs const &);
std::ostream &operator<<(std::ostream &, CastAttrs const &);
} // namespace FlexFlow

#endif // _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_CAST_ATTRS_DTG_H
