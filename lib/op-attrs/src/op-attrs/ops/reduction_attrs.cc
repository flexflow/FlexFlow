// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/op-attrs/include/op-attrs/ops/reduction_attrs.struct.toml

#include "op-attrs/ops/reduction_attrs.h"

namespace FlexFlow {
ReductionAttrs::ReductionAttrs(::FlexFlow::ff_dim_t const &reduction_dim,
                               int const &reduction_degree)
    : reduction_dim(reduction_dim), reduction_degree(reduction_degree) {}
bool ReductionAttrs::operator==(ReductionAttrs const &other) const {
  return std::tie(this->reduction_dim, this->reduction_degree) ==
         std::tie(other.reduction_dim, other.reduction_degree);
}
bool ReductionAttrs::operator!=(ReductionAttrs const &other) const {
  return std::tie(this->reduction_dim, this->reduction_degree) !=
         std::tie(other.reduction_dim, other.reduction_degree);
}
bool ReductionAttrs::operator<(ReductionAttrs const &other) const {
  return std::tie(this->reduction_dim, this->reduction_degree) <
         std::tie(other.reduction_dim, other.reduction_degree);
}
bool ReductionAttrs::operator>(ReductionAttrs const &other) const {
  return std::tie(this->reduction_dim, this->reduction_degree) >
         std::tie(other.reduction_dim, other.reduction_degree);
}
bool ReductionAttrs::operator<=(ReductionAttrs const &other) const {
  return std::tie(this->reduction_dim, this->reduction_degree) <=
         std::tie(other.reduction_dim, other.reduction_degree);
}
bool ReductionAttrs::operator>=(ReductionAttrs const &other) const {
  return std::tie(this->reduction_dim, this->reduction_degree) >=
         std::tie(other.reduction_dim, other.reduction_degree);
}
} // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::ReductionAttrs>::operator()(
    FlexFlow::ReductionAttrs const &x) const {
  size_t result = 0;
  result ^= std::hash<::FlexFlow::ff_dim_t>{}(x.reduction_dim) + 0x9e3779b9 +
            (result << 6) + (result >> 2);
  result ^= std::hash<int>{}(x.reduction_degree) + 0x9e3779b9 + (result << 6) +
            (result >> 2);
  return result;
}
} // namespace std

namespace nlohmann {
FlexFlow::ReductionAttrs
    adl_serializer<FlexFlow::ReductionAttrs>::from_json(json const &j) {
  return {j.at("reduction_dim").template get<::FlexFlow::ff_dim_t>(),
          j.at("reduction_degree").template get<int>()};
}
void adl_serializer<FlexFlow::ReductionAttrs>::to_json(
    json &j, FlexFlow::ReductionAttrs const &v) {
  j["__type"] = "ReductionAttrs";
  j["reduction_dim"] = v.reduction_dim;
  j["reduction_degree"] = v.reduction_degree;
}
} // namespace nlohmann

namespace rc {
Gen<FlexFlow::ReductionAttrs> Arbitrary<FlexFlow::ReductionAttrs>::arbitrary() {
  return gen::construct<FlexFlow::ReductionAttrs>(
      gen::arbitrary<::FlexFlow::ff_dim_t>(), gen::arbitrary<int>());
}
} // namespace rc

namespace FlexFlow {
std::string format_as(ReductionAttrs const &x) {
  std::ostringstream oss;
  oss << "<ReductionAttrs";
  oss << " reduction_dim=" << x.reduction_dim;
  oss << " reduction_degree=" << x.reduction_degree;
  oss << ">";
  return oss.str();
}
std::ostream &operator<<(std::ostream &s, ReductionAttrs const &x) {
  return s << fmt::to_string(x);
}
} // namespace FlexFlow
