// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/op-attrs/include/op-attrs/ops/replicate_attrs.struct.toml

#include "op-attrs/ops/replicate_attrs.h"

namespace FlexFlow {
ReplicateAttrs::ReplicateAttrs(::FlexFlow::ff_dim_t const &replicate_dim,
                               int const &replicate_degree)
    : replicate_dim(replicate_dim), replicate_degree(replicate_degree) {}
bool ReplicateAttrs::operator==(ReplicateAttrs const &other) const {
  return std::tie(this->replicate_dim, this->replicate_degree) ==
         std::tie(other.replicate_dim, other.replicate_degree);
}
bool ReplicateAttrs::operator!=(ReplicateAttrs const &other) const {
  return std::tie(this->replicate_dim, this->replicate_degree) !=
         std::tie(other.replicate_dim, other.replicate_degree);
}
bool ReplicateAttrs::operator<(ReplicateAttrs const &other) const {
  return std::tie(this->replicate_dim, this->replicate_degree) <
         std::tie(other.replicate_dim, other.replicate_degree);
}
bool ReplicateAttrs::operator>(ReplicateAttrs const &other) const {
  return std::tie(this->replicate_dim, this->replicate_degree) >
         std::tie(other.replicate_dim, other.replicate_degree);
}
bool ReplicateAttrs::operator<=(ReplicateAttrs const &other) const {
  return std::tie(this->replicate_dim, this->replicate_degree) <=
         std::tie(other.replicate_dim, other.replicate_degree);
}
bool ReplicateAttrs::operator>=(ReplicateAttrs const &other) const {
  return std::tie(this->replicate_dim, this->replicate_degree) >=
         std::tie(other.replicate_dim, other.replicate_degree);
}
} // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::ReplicateAttrs>::operator()(
    FlexFlow::ReplicateAttrs const &x) const {
  size_t result = 0;
  result ^= std::hash<::FlexFlow::ff_dim_t>{}(x.replicate_dim) + 0x9e3779b9 +
            (result << 6) + (result >> 2);
  result ^= std::hash<int>{}(x.replicate_degree) + 0x9e3779b9 + (result << 6) +
            (result >> 2);
  return result;
}
} // namespace std

namespace nlohmann {
FlexFlow::ReplicateAttrs
    adl_serializer<FlexFlow::ReplicateAttrs>::from_json(json const &j) {
  return {j.at("replicate_dim").template get<::FlexFlow::ff_dim_t>(),
          j.at("replicate_degree").template get<int>()};
}
void adl_serializer<FlexFlow::ReplicateAttrs>::to_json(
    json &j, FlexFlow::ReplicateAttrs const &v) {
  j["__type"] = "ReplicateAttrs";
  j["replicate_dim"] = v.replicate_dim;
  j["replicate_degree"] = v.replicate_degree;
}
} // namespace nlohmann

namespace rc {
Gen<FlexFlow::ReplicateAttrs> Arbitrary<FlexFlow::ReplicateAttrs>::arbitrary() {
  return gen::construct<FlexFlow::ReplicateAttrs>(
      gen::arbitrary<::FlexFlow::ff_dim_t>(), gen::arbitrary<int>());
}
} // namespace rc

namespace FlexFlow {
std::string format_as(ReplicateAttrs const &x) {
  std::ostringstream oss;
  oss << "<ReplicateAttrs";
  oss << " replicate_dim=" << x.replicate_dim;
  oss << " replicate_degree=" << x.replicate_degree;
  oss << ">";
  return oss.str();
}
std::ostream &operator<<(std::ostream &s, ReplicateAttrs const &x) {
  return s << fmt::to_string(x);
}
} // namespace FlexFlow
