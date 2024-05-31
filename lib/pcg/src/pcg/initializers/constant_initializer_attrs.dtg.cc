// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/pcg/include/pcg/initializers/constant_initializer_attrs.struct.toml
/* proj-data
{
  "generated_from": "0162b9c49fe6cbfc65410c6fa8dec427"
}
*/

#include "pcg/initializers/constant_initializer_attrs.dtg.h"

#include "op-attrs/datatype.h"
#include "utils/json.h"
#include <sstream>

namespace FlexFlow {
ConstantInitializerAttrs::ConstantInitializerAttrs(
    ::FlexFlow::DataTypeValue const &value)
    : value(value) {}
bool ConstantInitializerAttrs::operator==(
    ConstantInitializerAttrs const &other) const {
  return std::tie(this->value) == std::tie(other.value);
}
bool ConstantInitializerAttrs::operator!=(
    ConstantInitializerAttrs const &other) const {
  return std::tie(this->value) != std::tie(other.value);
}
bool ConstantInitializerAttrs::operator<(
    ConstantInitializerAttrs const &other) const {
  return std::tie(this->value) < std::tie(other.value);
}
bool ConstantInitializerAttrs::operator>(
    ConstantInitializerAttrs const &other) const {
  return std::tie(this->value) > std::tie(other.value);
}
bool ConstantInitializerAttrs::operator<=(
    ConstantInitializerAttrs const &other) const {
  return std::tie(this->value) <= std::tie(other.value);
}
bool ConstantInitializerAttrs::operator>=(
    ConstantInitializerAttrs const &other) const {
  return std::tie(this->value) >= std::tie(other.value);
}
} // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::ConstantInitializerAttrs>::operator()(
    FlexFlow::ConstantInitializerAttrs const &x) const {
  size_t result = 0;
  result ^= std::hash<::FlexFlow::DataTypeValue>{}(x.value) + 0x9e3779b9 +
            (result << 6) + (result >> 2);
  return result;
}
} // namespace std

namespace nlohmann {
FlexFlow::ConstantInitializerAttrs
    adl_serializer<FlexFlow::ConstantInitializerAttrs>::from_json(
        json const &j) {
  return {j.at("value").template get<::FlexFlow::DataTypeValue>()};
}
void adl_serializer<FlexFlow::ConstantInitializerAttrs>::to_json(
    json &j, FlexFlow::ConstantInitializerAttrs const &v) {
  j["__type"] = "ConstantInitializerAttrs";
  j["value"] = v.value;
}
} // namespace nlohmann

namespace FlexFlow {
std::string format_as(ConstantInitializerAttrs const &x) {
  std::ostringstream oss;
  oss << "<ConstantInitializerAttrs";
  oss << " value=" << x.value;
  oss << ">";
  return oss.str();
}
std::ostream &operator<<(std::ostream &s, ConstantInitializerAttrs const &x) {
  return s << fmt::to_string(x);
}
} // namespace FlexFlow