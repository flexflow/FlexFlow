// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/op-attrs/include/op-attrs/ops/split_attrs.struct.toml
/* proj-data
{
  "generated_from": "cde6b5caf6739d3b02fe8fce0d8ae8c5"
}
*/

#include "op-attrs/ops/split_attrs.dtg.h"

#include "op-attrs/ff_dim.dtg.h"
#include "op-attrs/ff_dim.h"
#include "utils/stack_vector.h"
#include <sstream>

namespace FlexFlow {
SplitAttrs::SplitAttrs(
    ::FlexFlow::stack_vector<int, MAX_NUM_OUTPUTS> const &splits,
    ::FlexFlow::ff_dim_t const &axis)
    : splits(splits), axis(axis) {}
bool SplitAttrs::operator==(SplitAttrs const &other) const {
  return std::tie(this->splits, this->axis) ==
         std::tie(other.splits, other.axis);
}
bool SplitAttrs::operator!=(SplitAttrs const &other) const {
  return std::tie(this->splits, this->axis) !=
         std::tie(other.splits, other.axis);
}
bool SplitAttrs::operator<(SplitAttrs const &other) const {
  return std::tie(this->splits, this->axis) <
         std::tie(other.splits, other.axis);
}
bool SplitAttrs::operator>(SplitAttrs const &other) const {
  return std::tie(this->splits, this->axis) >
         std::tie(other.splits, other.axis);
}
bool SplitAttrs::operator<=(SplitAttrs const &other) const {
  return std::tie(this->splits, this->axis) <=
         std::tie(other.splits, other.axis);
}
bool SplitAttrs::operator>=(SplitAttrs const &other) const {
  return std::tie(this->splits, this->axis) >=
         std::tie(other.splits, other.axis);
}
} // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::SplitAttrs>::operator()(
    FlexFlow::SplitAttrs const &x) const {
  size_t result = 0;
  result ^=
      std::hash<::FlexFlow::stack_vector<int, MAX_NUM_OUTPUTS>>{}(x.splits) +
      0x9e3779b9 + (result << 6) + (result >> 2);
  result ^= std::hash<::FlexFlow::ff_dim_t>{}(x.axis) + 0x9e3779b9 +
            (result << 6) + (result >> 2);
  return result;
}
} // namespace std

namespace nlohmann {
FlexFlow::SplitAttrs
    adl_serializer<FlexFlow::SplitAttrs>::from_json(json const &j) {
  return {j.at("splits")
              .template get<::FlexFlow::stack_vector<int, MAX_NUM_OUTPUTS>>(),
          j.at("axis").template get<::FlexFlow::ff_dim_t>()};
}
void adl_serializer<FlexFlow::SplitAttrs>::to_json(
    json &j, FlexFlow::SplitAttrs const &v) {
  j["__type"] = "SplitAttrs";
  j["splits"] = v.splits;
  j["axis"] = v.axis;
}
} // namespace nlohmann

namespace rc {
Gen<FlexFlow::SplitAttrs> Arbitrary<FlexFlow::SplitAttrs>::arbitrary() {
  return gen::construct<FlexFlow::SplitAttrs>(
      gen::arbitrary<::FlexFlow::stack_vector<int, MAX_NUM_OUTPUTS>>(),
      gen::arbitrary<::FlexFlow::ff_dim_t>());
}
} // namespace rc

namespace FlexFlow {
std::string format_as(SplitAttrs const &x) {
  std::ostringstream oss;
  oss << "<SplitAttrs";
  oss << " splits=" << x.splits;
  oss << " axis=" << x.axis;
  oss << ">";
  return oss.str();
}
std::ostream &operator<<(std::ostream &s, SplitAttrs const &x) {
  return s << fmt::to_string(x);
}
} // namespace FlexFlow
