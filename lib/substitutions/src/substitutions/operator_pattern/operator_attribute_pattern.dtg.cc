// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/substitutions/include/substitutions/operator_pattern/operator_attribute_pattern.struct.toml
/* proj-data
{
  "generated_from": "968d7a3e93303a7fa7482bbcd50246b6"
}
*/

#include "substitutions/operator_pattern/operator_attribute_pattern.dtg.h"

#include <sstream>

namespace FlexFlow {
OperatorAttributePattern::OperatorAttributePattern(
    std::unordered_set<::FlexFlow::OperatorAttributeConstraint> const
        &attribute_constraints)
    : attribute_constraints(attribute_constraints) {}
bool OperatorAttributePattern::operator==(
    OperatorAttributePattern const &other) const {
  return std::tie(this->attribute_constraints) ==
         std::tie(other.attribute_constraints);
}
bool OperatorAttributePattern::operator!=(
    OperatorAttributePattern const &other) const {
  return std::tie(this->attribute_constraints) !=
         std::tie(other.attribute_constraints);
}
} // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::OperatorAttributePattern>::operator()(
    ::FlexFlow::OperatorAttributePattern const &x) const {
  size_t result = 0;
  result ^=
      std::hash<std::unordered_set<::FlexFlow::OperatorAttributeConstraint>>{}(
          x.attribute_constraints) +
      0x9e3779b9 + (result << 6) + (result >> 2);
  return result;
}
} // namespace std

namespace nlohmann {
::FlexFlow::OperatorAttributePattern
    adl_serializer<::FlexFlow::OperatorAttributePattern>::from_json(
        json const &j) {
  return ::FlexFlow::OperatorAttributePattern{
      j.at("attribute_constraints")
          .template get<
              std::unordered_set<::FlexFlow::OperatorAttributeConstraint>>()};
}
void adl_serializer<::FlexFlow::OperatorAttributePattern>::to_json(
    json &j, ::FlexFlow::OperatorAttributePattern const &v) {
  j["__type"] = "OperatorAttributePattern";
  j["attribute_constraints"] = v.attribute_constraints;
}
} // namespace nlohmann

namespace FlexFlow {
std::string format_as(OperatorAttributePattern const &x) {
  std::ostringstream oss;
  oss << "<OperatorAttributePattern";
  oss << " attribute_constraints=" << x.attribute_constraints;
  oss << ">";
  return oss.str();
}
std::ostream &operator<<(std::ostream &s, OperatorAttributePattern const &x) {
  return s << fmt::to_string(x);
}
} // namespace FlexFlow
