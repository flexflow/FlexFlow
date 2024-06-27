// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/op-attrs/include/op-attrs/ops/element_binary_attrs.struct.toml
/* proj-data
{
  "generated_from": "2bb947c9cc92e3833ee88c908c539629"
}
*/

#include "op-attrs/ops/element_binary_attrs.dtg.h"

#include "op-attrs/datatype.h"
#include "op-attrs/operator_type.h"
#include <sstream>

namespace FlexFlow {
ElementBinaryAttrs::ElementBinaryAttrs(::FlexFlow::OperatorType const &type,
                                       ::FlexFlow::DataType const &compute_type,
                                       bool const &should_broadcast_lhs,
                                       bool const &should_broadcast_rhs)
    : type(type), compute_type(compute_type),
      should_broadcast_lhs(should_broadcast_lhs),
      should_broadcast_rhs(should_broadcast_rhs) {}
bool ElementBinaryAttrs::operator==(ElementBinaryAttrs const &other) const {
  return std::tie(this->type,
                  this->compute_type,
                  this->should_broadcast_lhs,
                  this->should_broadcast_rhs) ==
         std::tie(other.type,
                  other.compute_type,
                  other.should_broadcast_lhs,
                  other.should_broadcast_rhs);
}
bool ElementBinaryAttrs::operator!=(ElementBinaryAttrs const &other) const {
  return std::tie(this->type,
                  this->compute_type,
                  this->should_broadcast_lhs,
                  this->should_broadcast_rhs) !=
         std::tie(other.type,
                  other.compute_type,
                  other.should_broadcast_lhs,
                  other.should_broadcast_rhs);
}
bool ElementBinaryAttrs::operator<(ElementBinaryAttrs const &other) const {
  return std::tie(this->type,
                  this->compute_type,
                  this->should_broadcast_lhs,
                  this->should_broadcast_rhs) <
         std::tie(other.type,
                  other.compute_type,
                  other.should_broadcast_lhs,
                  other.should_broadcast_rhs);
}
bool ElementBinaryAttrs::operator>(ElementBinaryAttrs const &other) const {
  return std::tie(this->type,
                  this->compute_type,
                  this->should_broadcast_lhs,
                  this->should_broadcast_rhs) >
         std::tie(other.type,
                  other.compute_type,
                  other.should_broadcast_lhs,
                  other.should_broadcast_rhs);
}
bool ElementBinaryAttrs::operator<=(ElementBinaryAttrs const &other) const {
  return std::tie(this->type,
                  this->compute_type,
                  this->should_broadcast_lhs,
                  this->should_broadcast_rhs) <=
         std::tie(other.type,
                  other.compute_type,
                  other.should_broadcast_lhs,
                  other.should_broadcast_rhs);
}
bool ElementBinaryAttrs::operator>=(ElementBinaryAttrs const &other) const {
  return std::tie(this->type,
                  this->compute_type,
                  this->should_broadcast_lhs,
                  this->should_broadcast_rhs) >=
         std::tie(other.type,
                  other.compute_type,
                  other.should_broadcast_lhs,
                  other.should_broadcast_rhs);
}
} // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::ElementBinaryAttrs>::operator()(
    ::FlexFlow::ElementBinaryAttrs const &x) const {
  size_t result = 0;
  result ^= std::hash<::FlexFlow::OperatorType>{}(x.type) + 0x9e3779b9 +
            (result << 6) + (result >> 2);
  result ^= std::hash<::FlexFlow::DataType>{}(x.compute_type) + 0x9e3779b9 +
            (result << 6) + (result >> 2);
  result ^= std::hash<bool>{}(x.should_broadcast_lhs) + 0x9e3779b9 +
            (result << 6) + (result >> 2);
  result ^= std::hash<bool>{}(x.should_broadcast_rhs) + 0x9e3779b9 +
            (result << 6) + (result >> 2);
  return result;
}
} // namespace std

namespace nlohmann {
::FlexFlow::ElementBinaryAttrs
    adl_serializer<::FlexFlow::ElementBinaryAttrs>::from_json(json const &j) {
  return ::FlexFlow::ElementBinaryAttrs{
      j.at("type").template get<::FlexFlow::OperatorType>(),
      j.at("compute_type").template get<::FlexFlow::DataType>(),
      j.at("should_broadcast_lhs").template get<bool>(),
      j.at("should_broadcast_rhs").template get<bool>()};
}
void adl_serializer<::FlexFlow::ElementBinaryAttrs>::to_json(
    json &j, ::FlexFlow::ElementBinaryAttrs const &v) {
  j["__type"] = "ElementBinaryAttrs";
  j["type"] = v.type;
  j["compute_type"] = v.compute_type;
  j["should_broadcast_lhs"] = v.should_broadcast_lhs;
  j["should_broadcast_rhs"] = v.should_broadcast_rhs;
}
} // namespace nlohmann

namespace rc {
Gen<::FlexFlow::ElementBinaryAttrs>
    Arbitrary<::FlexFlow::ElementBinaryAttrs>::arbitrary() {
  return gen::construct<::FlexFlow::ElementBinaryAttrs>(
      gen::arbitrary<::FlexFlow::OperatorType>(),
      gen::arbitrary<::FlexFlow::DataType>(),
      gen::arbitrary<bool>(),
      gen::arbitrary<bool>());
}
} // namespace rc

namespace FlexFlow {
std::string format_as(ElementBinaryAttrs const &x) {
  std::ostringstream oss;
  oss << "<ElementBinaryAttrs";
  oss << " type=" << x.type;
  oss << " compute_type=" << x.compute_type;
  oss << " should_broadcast_lhs=" << x.should_broadcast_lhs;
  oss << " should_broadcast_rhs=" << x.should_broadcast_rhs;
  oss << ">";
  return oss.str();
}
std::ostream &operator<<(std::ostream &s, ElementBinaryAttrs const &x) {
  return s << fmt::to_string(x);
}
} // namespace FlexFlow
