// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/op-attrs/include/op-attrs/ops/conv_2d/conv_2d_input_shape.struct.toml
/* proj-data
{
  "generated_from": "51911f58c134d55b2d0245444acbae53"
}
*/

#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_CONV_2D_CONV_2D_INPUT_SHAPE_DTG_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_CONV_2D_CONV_2D_INPUT_SHAPE_DTG_H

#include "fmt/format.h"
#include "nlohmann/json.hpp"
#include "op-attrs/datatype.dtg.h"
#include "rapidcheck.h"
#include <cstddef>
#include <functional>
#include <ostream>
#include <tuple>

namespace FlexFlow {
struct Conv2DInputShape {
  Conv2DInputShape() = delete;
  Conv2DInputShape(size_t const &num_samples,
                   size_t const &num_channels,
                   size_t const &height,
                   size_t const &width,
                   ::FlexFlow::DataType const &datatype);

  bool operator==(Conv2DInputShape const &) const;
  bool operator!=(Conv2DInputShape const &) const;
  bool operator<(Conv2DInputShape const &) const;
  bool operator>(Conv2DInputShape const &) const;
  bool operator<=(Conv2DInputShape const &) const;
  bool operator>=(Conv2DInputShape const &) const;
  size_t num_samples;
  size_t num_channels;
  size_t height;
  size_t width;
  ::FlexFlow::DataType datatype;
};
} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::Conv2DInputShape> {
  size_t operator()(FlexFlow::Conv2DInputShape const &) const;
};
} // namespace std

namespace nlohmann {
template <>
struct adl_serializer<FlexFlow::Conv2DInputShape> {
  static FlexFlow::Conv2DInputShape from_json(json const &);
  static void to_json(json &, FlexFlow::Conv2DInputShape const &);
};
} // namespace nlohmann

namespace rc {
template <>
struct Arbitrary<FlexFlow::Conv2DInputShape> {
  static Gen<FlexFlow::Conv2DInputShape> arbitrary();
};
} // namespace rc

namespace FlexFlow {
std::string format_as(Conv2DInputShape const &);
std::ostream &operator<<(std::ostream &, Conv2DInputShape const &);
} // namespace FlexFlow

#endif // _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_CONV_2D_CONV_2D_INPUT_SHAPE_DTG_H
