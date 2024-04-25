// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/op-attrs/include/op-attrs/ops/cast_attrs.struct.toml

#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_CAST_ATTRS_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_CAST_ATTRS_H

#include "fmt/format.h"
#include "nlohmann/json.hpp"
#include "op-attrs/datatype.h"
#include "rapidcheck.h"
#include <functional>
#include <ostream>
#include <sstream>
#include <tuple>

namespace FlexFlow {
struct CastAttrs {
  CastAttrs() = delete;
  CastAttrs(DataType const &dtype);

  bool operator==(CastAttrs const &) const;
  bool operator!=(CastAttrs const &) const;
  bool operator<(CastAttrs const &) const;
  bool operator>(CastAttrs const &) const;
  bool operator<=(CastAttrs const &) const;
  bool operator>=(CastAttrs const &) const;
  DataType dtype;
};
} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::CastAttrs> {
  size_t operator()(FlexFlow::CastAttrs const &) const;
};
} // namespace std

namespace nlohmann {
template <>
struct adl_serializer<FlexFlow::CastAttrs> {
  static FlexFlow::CastAttrs from_json(json const &);
  static void to_json(json &, FlexFlow::CastAttrs const &);
};
} // namespace nlohmann

namespace rc {
template <>
struct Arbitrary<FlexFlow::CastAttrs> {
  static Gen<FlexFlow::CastAttrs> arbitrary();
};
} // namespace rc

namespace FlexFlow {
std::string format_as(CastAttrs const &);
std::ostream &operator<<(std::ostream &, CastAttrs const &);
} // namespace FlexFlow

#endif // _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_CAST_ATTRS_H
