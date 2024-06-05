// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/op-attrs/include/op-attrs/ops/combine_attrs.struct.toml
/* proj-data
{
  "generated_from": "58fc5a388fd1a325ef4142094607e39a"
}
*/

#include "op-attrs/ops/combine_attrs.dtg.h"

#include "op-attrs/ff_dim.dtg.h"
#include "op-attrs/ff_dim.h"
#include <sstream>

namespace FlexFlow {
CombineAttrs::CombineAttrs(::FlexFlow::ff_dim_t const &combine_dim,
                           int const &combine_degree)
    : combine_dim(combine_dim), combine_degree(combine_degree) {}
bool CombineAttrs::operator==(CombineAttrs const &other) const {
  return std::tie(this->combine_dim, this->combine_degree) ==
         std::tie(other.combine_dim, other.combine_degree);
}
bool CombineAttrs::operator!=(CombineAttrs const &other) const {
  return std::tie(this->combine_dim, this->combine_degree) !=
         std::tie(other.combine_dim, other.combine_degree);
}
bool CombineAttrs::operator<(CombineAttrs const &other) const {
  return std::tie(this->combine_dim, this->combine_degree) <
         std::tie(other.combine_dim, other.combine_degree);
}
bool CombineAttrs::operator>(CombineAttrs const &other) const {
  return std::tie(this->combine_dim, this->combine_degree) >
         std::tie(other.combine_dim, other.combine_degree);
}
bool CombineAttrs::operator<=(CombineAttrs const &other) const {
  return std::tie(this->combine_dim, this->combine_degree) <=
         std::tie(other.combine_dim, other.combine_degree);
}
bool CombineAttrs::operator>=(CombineAttrs const &other) const {
  return std::tie(this->combine_dim, this->combine_degree) >=
         std::tie(other.combine_dim, other.combine_degree);
}
} // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::CombineAttrs>::operator()(
    FlexFlow::CombineAttrs const &x) const {
  size_t result = 0;
  result ^= std::hash<::FlexFlow::ff_dim_t>{}(x.combine_dim) + 0x9e3779b9 +
            (result << 6) + (result >> 2);
  result ^= std::hash<int>{}(x.combine_degree) + 0x9e3779b9 + (result << 6) +
            (result >> 2);
  return result;
}
} // namespace std

namespace nlohmann {
FlexFlow::CombineAttrs
    adl_serializer<FlexFlow::CombineAttrs>::from_json(json const &j) {
  return {j.at("combine_dim").template get<::FlexFlow::ff_dim_t>(),
          j.at("combine_degree").template get<int>()};
}
void adl_serializer<FlexFlow::CombineAttrs>::to_json(
    json &j, FlexFlow::CombineAttrs const &v) {
  j["__type"] = "CombineAttrs";
  j["combine_dim"] = v.combine_dim;
  j["combine_degree"] = v.combine_degree;
}
} // namespace nlohmann

namespace rc {
Gen<FlexFlow::CombineAttrs> Arbitrary<FlexFlow::CombineAttrs>::arbitrary() {
  return gen::construct<FlexFlow::CombineAttrs>(
      gen::arbitrary<::FlexFlow::ff_dim_t>(), gen::arbitrary<int>());
}
} // namespace rc

namespace FlexFlow {
std::string format_as(CombineAttrs const &x) {
  std::ostringstream oss;
  oss << "<CombineAttrs";
  oss << " combine_dim=" << x.combine_dim;
  oss << " combine_degree=" << x.combine_degree;
  oss << ">";
  return oss.str();
}
std::ostream &operator<<(std::ostream &s, CombineAttrs const &x) {
  return s << fmt::to_string(x);
}
} // namespace FlexFlow
