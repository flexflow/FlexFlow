// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/pcg/include/pcg/layer_attrs.struct.toml
/* proj-data
{
  "generated_from": "b3e4f0c07a906139b599bd4696cb5e65"
}
*/

#include "pcg/layer_attrs.dtg.h"

#include <sstream>

namespace FlexFlow {
LayerAttrs::LayerAttrs(
    ::FlexFlow::ComputationGraphOpAttrs const &attrs,
    std::optional<::FlexFlow::stack_string<MAX_OPNAME>> const &name)
    : attrs(attrs), name(name) {}
bool LayerAttrs::operator==(LayerAttrs const &other) const {
  return std::tie(this->attrs, this->name) == std::tie(other.attrs, other.name);
}
bool LayerAttrs::operator!=(LayerAttrs const &other) const {
  return std::tie(this->attrs, this->name) != std::tie(other.attrs, other.name);
}
bool LayerAttrs::operator<(LayerAttrs const &other) const {
  return std::tie(this->attrs, this->name) < std::tie(other.attrs, other.name);
}
bool LayerAttrs::operator>(LayerAttrs const &other) const {
  return std::tie(this->attrs, this->name) > std::tie(other.attrs, other.name);
}
bool LayerAttrs::operator<=(LayerAttrs const &other) const {
  return std::tie(this->attrs, this->name) <= std::tie(other.attrs, other.name);
}
bool LayerAttrs::operator>=(LayerAttrs const &other) const {
  return std::tie(this->attrs, this->name) >= std::tie(other.attrs, other.name);
}
} // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::LayerAttrs>::operator()(
    ::FlexFlow::LayerAttrs const &x) const {
  size_t result = 0;
  result ^= std::hash<::FlexFlow::ComputationGraphOpAttrs>{}(x.attrs) +
            0x9e3779b9 + (result << 6) + (result >> 2);
  result ^=
      std::hash<std::optional<::FlexFlow::stack_string<MAX_OPNAME>>>{}(x.name) +
      0x9e3779b9 + (result << 6) + (result >> 2);
  return result;
}
} // namespace std

namespace nlohmann {
::FlexFlow::LayerAttrs
    adl_serializer<::FlexFlow::LayerAttrs>::from_json(json const &j) {
  return ::FlexFlow::LayerAttrs{
      j.at("attrs").template get<::FlexFlow::ComputationGraphOpAttrs>(),
      j.at("name")
          .template get<std::optional<::FlexFlow::stack_string<MAX_OPNAME>>>()};
}
void adl_serializer<::FlexFlow::LayerAttrs>::to_json(
    json &j, ::FlexFlow::LayerAttrs const &v) {
  j["__type"] = "LayerAttrs";
  j["attrs"] = v.attrs;
  j["name"] = v.name;
}
} // namespace nlohmann

namespace FlexFlow {
std::string format_as(LayerAttrs const &x) {
  std::ostringstream oss;
  oss << "<LayerAttrs";
  oss << " attrs=" << x.attrs;
  oss << " name=" << x.name;
  oss << ">";
  return oss.str();
}
std::ostream &operator<<(std::ostream &s, LayerAttrs const &x) {
  return s << fmt::to_string(x);
}
} // namespace FlexFlow
