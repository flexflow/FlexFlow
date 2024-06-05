// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/op-attrs/include/op-attrs/tensor_dims.struct.toml
/* proj-data
{
  "generated_from": "5beb89eeae9eba303f90e726c794375d"
}
*/

#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_TENSOR_DIMS_DTG_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_TENSOR_DIMS_DTG_H

#include "fmt/format.h"
#include "nlohmann/json.hpp"
#include "op-attrs/dim_ordered.h"
#include "rapidcheck.h"
#include <functional>
#include <ostream>
#include <tuple>

namespace FlexFlow {
struct TensorDims {
  TensorDims() = delete;
  TensorDims(::FlexFlow::FFOrdered<size_t> const &ff_ordered);

  bool operator==(TensorDims const &) const;
  bool operator!=(TensorDims const &) const;
  bool operator<(TensorDims const &) const;
  bool operator>(TensorDims const &) const;
  bool operator<=(TensorDims const &) const;
  bool operator>=(TensorDims const &) const;
  ::FlexFlow::FFOrdered<size_t> ff_ordered;
};
} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::TensorDims> {
  size_t operator()(FlexFlow::TensorDims const &) const;
};
} // namespace std

namespace nlohmann {
template <>
struct adl_serializer<FlexFlow::TensorDims> {
  static FlexFlow::TensorDims from_json(json const &);
  static void to_json(json &, FlexFlow::TensorDims const &);
};
} // namespace nlohmann

namespace rc {
template <>
struct Arbitrary<FlexFlow::TensorDims> {
  static Gen<FlexFlow::TensorDims> arbitrary();
};
} // namespace rc

namespace FlexFlow {
std::string format_as(TensorDims const &);
std::ostream &operator<<(std::ostream &, TensorDims const &);
} // namespace FlexFlow

#endif // _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_TENSOR_DIMS_DTG_H
