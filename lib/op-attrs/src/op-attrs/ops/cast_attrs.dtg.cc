// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/op-attrs/include/op-attrs/ops/cast_attrs.struct.toml
/* proj-data
{
  "generated_from": "c171c87db89b9ec9ea7d52a50c153054"
}
*/

#include "op-attrs/ops/cast_attrs.dtg.h"

#include <sstream>

namespace FlexFlow {
CastAttrs::CastAttrs(DataType const &dtype) : dtype(dtype) {}
bool CastAttrs::operator==(CastAttrs const &other) const {
  return std::tie(this->dtype) == std::tie(other.dtype);
}
bool CastAttrs::operator!=(CastAttrs const &other) const {
  return std::tie(this->dtype) != std::tie(other.dtype);
}
bool CastAttrs::operator<(CastAttrs const &other) const {
  return std::tie(this->dtype) < std::tie(other.dtype);
}
bool CastAttrs::operator>(CastAttrs const &other) const {
  return std::tie(this->dtype) > std::tie(other.dtype);
}
bool CastAttrs::operator<=(CastAttrs const &other) const {
  return std::tie(this->dtype) <= std::tie(other.dtype);
}
bool CastAttrs::operator>=(CastAttrs const &other) const {
  return std::tie(this->dtype) >= std::tie(other.dtype);
}
} // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::CastAttrs>::operator()(
    ::FlexFlow::CastAttrs const &x) const {
  size_t result = 0;
  result ^= std::hash<DataType>{}(x.dtype) + 0x9e3779b9 + (result << 6) +
            (result >> 2);
  return result;
}
} // namespace std

namespace nlohmann {
::FlexFlow::CastAttrs
    adl_serializer<::FlexFlow::CastAttrs>::from_json(json const &j) {
  return ::FlexFlow::CastAttrs{j.at("dtype").template get<DataType>()};
}
void adl_serializer<::FlexFlow::CastAttrs>::to_json(
    json &j, ::FlexFlow::CastAttrs const &v) {
  j["__type"] = "CastAttrs";
  j["dtype"] = v.dtype;
}
} // namespace nlohmann

namespace rc {
Gen<::FlexFlow::CastAttrs> Arbitrary<::FlexFlow::CastAttrs>::arbitrary() {
  return gen::construct<::FlexFlow::CastAttrs>(gen::arbitrary<DataType>());
}
} // namespace rc

namespace FlexFlow {
std::string format_as(CastAttrs const &x) {
  std::ostringstream oss;
  oss << "<CastAttrs";
  oss << " dtype=" << x.dtype;
  oss << ">";
  return oss.str();
}
std::ostream &operator<<(std::ostream &s, CastAttrs const &x) {
  return s << fmt::to_string(x);
}
} // namespace FlexFlow
