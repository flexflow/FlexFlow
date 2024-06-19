// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/op-attrs/include/op-attrs/ops/batch_norm_attrs.struct.toml
/* proj-data
{
  "generated_from": "f8e0219d8a3e008a73c38cf84d25f66e"
}
*/

#include "op-attrs/ops/batch_norm_attrs.dtg.h"

#include <sstream>

namespace FlexFlow {
BatchNormAttrs::BatchNormAttrs(bool const &relu) : relu(relu) {}
bool BatchNormAttrs::operator==(BatchNormAttrs const &other) const {
  return std::tie(this->relu) == std::tie(other.relu);
}
bool BatchNormAttrs::operator!=(BatchNormAttrs const &other) const {
  return std::tie(this->relu) != std::tie(other.relu);
}
bool BatchNormAttrs::operator<(BatchNormAttrs const &other) const {
  return std::tie(this->relu) < std::tie(other.relu);
}
bool BatchNormAttrs::operator>(BatchNormAttrs const &other) const {
  return std::tie(this->relu) > std::tie(other.relu);
}
bool BatchNormAttrs::operator<=(BatchNormAttrs const &other) const {
  return std::tie(this->relu) <= std::tie(other.relu);
}
bool BatchNormAttrs::operator>=(BatchNormAttrs const &other) const {
  return std::tie(this->relu) >= std::tie(other.relu);
}
} // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::BatchNormAttrs>::operator()(
    FlexFlow::BatchNormAttrs const &x) const {
  size_t result = 0;
  result ^=
      std::hash<bool>{}(x.relu) + 0x9e3779b9 + (result << 6) + (result >> 2);
  return result;
}
} // namespace std

namespace nlohmann {
FlexFlow::BatchNormAttrs
    adl_serializer<FlexFlow::BatchNormAttrs>::from_json(json const &j) {
  return {j.at("relu").template get<bool>()};
}
void adl_serializer<FlexFlow::BatchNormAttrs>::to_json(
    json &j, FlexFlow::BatchNormAttrs const &v) {
  j["__type"] = "BatchNormAttrs";
  j["relu"] = v.relu;
}
} // namespace nlohmann

namespace rc {
Gen<FlexFlow::BatchNormAttrs> Arbitrary<FlexFlow::BatchNormAttrs>::arbitrary() {
  return gen::construct<FlexFlow::BatchNormAttrs>(gen::arbitrary<bool>());
}
} // namespace rc

namespace FlexFlow {
std::string format_as(BatchNormAttrs const &x) {
  std::ostringstream oss;
  oss << "<BatchNormAttrs";
  oss << " relu=" << x.relu;
  oss << ">";
  return oss.str();
}
std::ostream &operator<<(std::ostream &s, BatchNormAttrs const &x) {
  return s << fmt::to_string(x);
}
} // namespace FlexFlow
