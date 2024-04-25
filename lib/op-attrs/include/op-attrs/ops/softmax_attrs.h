// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/op-attrs/include/op-attrs/ops/softmax_attrs.struct.toml

#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_SOFTMAX_ATTRS_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_SOFTMAX_ATTRS_H

#include "fmt/format.h"
#include "nlohmann/json.hpp"
#include "op-attrs/ff_dim.h"
#include "rapidcheck.h"
#include <functional>
#include <ostream>
#include <sstream>
#include <tuple>

namespace FlexFlow {
struct SoftmaxAttrs {
  SoftmaxAttrs() = delete;
  SoftmaxAttrs(::FlexFlow::ff_dim_t const &dim);

  bool operator==(SoftmaxAttrs const &) const;
  bool operator!=(SoftmaxAttrs const &) const;
  bool operator<(SoftmaxAttrs const &) const;
  bool operator>(SoftmaxAttrs const &) const;
  bool operator<=(SoftmaxAttrs const &) const;
  bool operator>=(SoftmaxAttrs const &) const;
  ::FlexFlow::ff_dim_t dim;
};
} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::SoftmaxAttrs> {
  size_t operator()(FlexFlow::SoftmaxAttrs const &) const;
};
} // namespace std

namespace nlohmann {
template <>
struct adl_serializer<FlexFlow::SoftmaxAttrs> {
  static FlexFlow::SoftmaxAttrs from_json(json const &);
  static void to_json(json &, FlexFlow::SoftmaxAttrs const &);
};
} // namespace nlohmann

namespace rc {
template <>
struct Arbitrary<FlexFlow::SoftmaxAttrs> {
  static Gen<FlexFlow::SoftmaxAttrs> arbitrary();
};
} // namespace rc

namespace FlexFlow {
std::string format_as(SoftmaxAttrs const &);
std::ostream &operator<<(std::ostream &, SoftmaxAttrs const &);
} // namespace FlexFlow

#endif // _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_SOFTMAX_ATTRS_H
