// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/pcg/include/pcg/strided_rectangle_side.struct.toml
/* proj-data
{
  "generated_from": "b14fcf1e28c262d22b92fac691ede3d4"
}
*/

#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_STRIDED_RECTANGLE_SIDE_DTG_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_STRIDED_RECTANGLE_SIDE_DTG_H

#include "fmt/format.h"
#include "nlohmann/json.hpp"
#include "pcg/num_points_t.dtg.h"
#include "rapidcheck.h"
#include <functional>
#include <ostream>
#include <tuple>

namespace FlexFlow {
struct StridedRectangleSide {
  StridedRectangleSide() = delete;
  StridedRectangleSide(::FlexFlow::num_points_t const &num_points,
                       int const &stride);

  bool operator==(StridedRectangleSide const &) const;
  bool operator!=(StridedRectangleSide const &) const;
  bool operator<(StridedRectangleSide const &) const;
  bool operator>(StridedRectangleSide const &) const;
  bool operator<=(StridedRectangleSide const &) const;
  bool operator>=(StridedRectangleSide const &) const;
  ::FlexFlow::num_points_t num_points;
  int stride;
};
} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::StridedRectangleSide> {
  size_t operator()(FlexFlow::StridedRectangleSide const &) const;
};
} // namespace std

namespace nlohmann {
template <>
struct adl_serializer<FlexFlow::StridedRectangleSide> {
  static FlexFlow::StridedRectangleSide from_json(json const &);
  static void to_json(json &, FlexFlow::StridedRectangleSide const &);
};
} // namespace nlohmann

namespace rc {
template <>
struct Arbitrary<FlexFlow::StridedRectangleSide> {
  static Gen<FlexFlow::StridedRectangleSide> arbitrary();
};
} // namespace rc

namespace FlexFlow {
std::string format_as(StridedRectangleSide const &);
std::ostream &operator<<(std::ostream &, StridedRectangleSide const &);
} // namespace FlexFlow

#endif // _FLEXFLOW_LIB_PCG_INCLUDE_PCG_STRIDED_RECTANGLE_SIDE_DTG_H