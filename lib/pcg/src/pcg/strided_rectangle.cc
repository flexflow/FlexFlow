#include "pcg/strided_rectangle.h"
#include "op-attrs/dim_ordered/transform.h"
#include "pcg/device_coordinates.dtg.h"
#include "pcg/device_id_t.dtg.h"
#include "pcg/strided_rectangle_side.dtg.h"
#include "pcg/strided_rectangle_side.h"
#include "utils/containers/as_vector.h"
#include "utils/containers/product.h"
#include "utils/containers/sorted.h"
#include "utils/containers/transform.h"
#include "utils/fmt/vector.h"
#include "utils/hash-utils.h"
#include "utils/hash/tuple.h"
#include "utils/hash/vector.h"

namespace FlexFlow {

StridedRectangle::StridedRectangle(
    std::vector<::FlexFlow::StridedRectangleSide> const &sides)
    : sides(sorted(sides)) {}

std::tuple<std::vector<StridedRectangleSide> const &>
    StridedRectangle::tie() const {
  return std::tie(sides);
}

bool StridedRectangle::operator==(StridedRectangle const &other) const {
  return this->tie() == other.tie();
}

bool StridedRectangle::operator!=(StridedRectangle const &other) const {
  return this->tie() != other.tie();
}

bool StridedRectangle::operator<(StridedRectangle const &other) const {
  return this->tie() < other.tie();
}

bool StridedRectangle::operator>(StridedRectangle const &other) const {
  return this->tie() > other.tie();
}

bool StridedRectangle::operator<=(StridedRectangle const &other) const {
  return this->tie() <= other.tie();
}

bool StridedRectangle::operator>=(StridedRectangle const &other) const {
  return this->tie() >= other.tie();
}

std::vector<StridedRectangleSide> const &StridedRectangle::get_sides() const {
  return sides;
}

StridedRectangleSide const &StridedRectangle::at(int idx) const {
  return this->sides.at(idx);
}

std::string format_as(StridedRectangle const &x) {
  std::ostringstream oss;
  oss << "<StridedRectangle";
  oss << " sides=" << x.get_sides();
  oss << ">";
  return oss.str();
}

std::ostream &operator<<(std::ostream &s, StridedRectangle const &x) {
  return s << fmt::to_string(x);
}

size_t get_num_dims(StridedRectangle const &rect) {
  return rect.get_sides().size();
}

num_points_t get_num_points(StridedRectangle const &rect) {
  return num_points_t{
      product(transform(rect.get_sides(), [](StridedRectangleSide const &side) {
        return side.num_points.unwrapped;
      }))};
}

size_t get_size(StridedRectangle const &rect) {
  return product(
      transform(rect.get_sides(), [](StridedRectangleSide const &side) {
        return get_side_size(side).unwrapped;
      }));
}

} // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::StridedRectangle>::operator()(
    ::FlexFlow::StridedRectangle const &x) const {
  return get_std_hash(x.tie());
}
} // namespace std

namespace nlohmann {
::FlexFlow::StridedRectangle
    adl_serializer<::FlexFlow::StridedRectangle>::from_json(json const &j) {
  return ::FlexFlow::StridedRectangle{
      j.at("sides")
          .template get<std::vector<::FlexFlow::StridedRectangleSide>>()};
}
void adl_serializer<::FlexFlow::StridedRectangle>::to_json(
    json &j, ::FlexFlow::StridedRectangle const &v) {
  j["__type"] = "StridedRectangle";
  j["sides"] = v.get_sides();
}
} // namespace nlohmann

namespace rc {
Gen<::FlexFlow::StridedRectangle>
    Arbitrary<::FlexFlow::StridedRectangle>::arbitrary() {
  return gen::construct<::FlexFlow::StridedRectangle>(
      gen::arbitrary<std::vector<::FlexFlow::StridedRectangleSide>>());
}
} // namespace rc
