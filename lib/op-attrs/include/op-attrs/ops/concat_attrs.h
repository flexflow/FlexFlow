// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/op-attrs/include/op-attrs/ops/concat_attrs.struct.toml

#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_CONCAT_ATTRS_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_CONCAT_ATTRS_H

#include "fmt/format.h"
#include "nlohmann/json.hpp"
#include "op-attrs/ff_dim.h"
#include "rapidcheck.h"
#include <functional>
#include <ostream>
#include <sstream>
#include <tuple>

namespace FlexFlow {
struct ConcatAttrs {
  ConcatAttrs() = delete;
  ConcatAttrs(::FlexFlow::ff_dim_t const &axis, int const &num_inputs);

  bool operator==(ConcatAttrs const &) const;
  bool operator!=(ConcatAttrs const &) const;
  bool operator<(ConcatAttrs const &) const;
  bool operator>(ConcatAttrs const &) const;
  bool operator<=(ConcatAttrs const &) const;
  bool operator>=(ConcatAttrs const &) const;
  ::FlexFlow::ff_dim_t axis;
  int num_inputs;
};
} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::ConcatAttrs> {
  size_t operator()(FlexFlow::ConcatAttrs const &) const;
};
} // namespace std

namespace nlohmann {
template <>
struct adl_serializer<FlexFlow::ConcatAttrs> {
  static FlexFlow::ConcatAttrs from_json(json const &);
  static void to_json(json &, FlexFlow::ConcatAttrs const &);
};
} // namespace nlohmann

namespace rc {
template <>
struct Arbitrary<FlexFlow::ConcatAttrs> {
  static Gen<FlexFlow::ConcatAttrs> arbitrary();
};
} // namespace rc

namespace FlexFlow {
std::string format_as(ConcatAttrs const &);
std::ostream &operator<<(std::ostream &, ConcatAttrs const &);
} // namespace FlexFlow

#endif // _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_CONCAT_ATTRS_H
