#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_STRIDED_RECTANGLE_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_STRIDED_RECTANGLE_H

#include "op-attrs/ff_dim.dtg.h"
#include "pcg/device_id_t.dtg.h"
#include "pcg/num_points_t.dtg.h"
#include "pcg/side_size_t.dtg.h"
#include "pcg/strided_rectangle_side.dtg.h"

namespace FlexFlow {

struct StridedRectangle {

private:
  std::vector<StridedRectangleSide> sides;
  std::tuple<std::vector<StridedRectangleSide> const &> tie() const;
  friend struct std::hash<StridedRectangle>;

public:
  StridedRectangle() = delete;
  explicit StridedRectangle(std::vector<StridedRectangleSide> const &sides);

  bool operator==(StridedRectangle const &) const;
  bool operator!=(StridedRectangle const &) const;
  bool operator<(StridedRectangle const &) const;
  bool operator>(StridedRectangle const &) const;
  bool operator<=(StridedRectangle const &) const;
  bool operator>=(StridedRectangle const &) const;

  StridedRectangleSide const &at(int idx) const;
  std::vector<StridedRectangleSide> const &get_sides() const;
};
std::string format_as(StridedRectangle const &);
std::ostream &operator<<(std::ostream &, StridedRectangle const &);

size_t get_num_dims(StridedRectangle const &rect);

num_points_t get_num_points(StridedRectangle const &rect);

} // namespace FlexFlow

namespace std {
template <>
struct hash<::FlexFlow::StridedRectangle> {
  size_t operator()(::FlexFlow::StridedRectangle const &) const;
};
} // namespace std

namespace nlohmann {
template <>
struct adl_serializer<::FlexFlow::StridedRectangle> {
  static ::FlexFlow::StridedRectangle from_json(json const &);
  static void to_json(json &, ::FlexFlow::StridedRectangle const &);
};
} // namespace nlohmann

namespace rc {
template <>
struct Arbitrary<::FlexFlow::StridedRectangle> {
  static Gen<::FlexFlow::StridedRectangle> arbitrary();
};
} // namespace rc

#endif
