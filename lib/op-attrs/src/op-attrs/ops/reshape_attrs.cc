// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/op-attrs/include/op-attrs/ops/reshape_attrs.struct.toml

#include "op-attrs/ops/reshape_attrs.h"

namespace FlexFlow {
ReshapeAttrs::ReshapeAttrs(::FlexFlow::TensorShape const &shape)
    : shape(shape) {}
bool ReshapeAttrs::operator==(ReshapeAttrs const &other) const {
  return std::tie(this->shape) == std::tie(other.shape);
}
bool ReshapeAttrs::operator!=(ReshapeAttrs const &other) const {
  return std::tie(this->shape) != std::tie(other.shape);
}
bool ReshapeAttrs::operator<(ReshapeAttrs const &other) const {
  return std::tie(this->shape) < std::tie(other.shape);
}
bool ReshapeAttrs::operator>(ReshapeAttrs const &other) const {
  return std::tie(this->shape) > std::tie(other.shape);
}
bool ReshapeAttrs::operator<=(ReshapeAttrs const &other) const {
  return std::tie(this->shape) <= std::tie(other.shape);
}
bool ReshapeAttrs::operator>=(ReshapeAttrs const &other) const {
  return std::tie(this->shape) >= std::tie(other.shape);
}
} // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::ReshapeAttrs>::operator()(
    FlexFlow::ReshapeAttrs const &x) const {
  size_t result = 0;
  result ^= std::hash<::FlexFlow::TensorShape>{}(x.shape) + 0x9e3779b9 +
            (result << 6) + (result >> 2);
  return result;
}
} // namespace std

namespace nlohmann {
FlexFlow::ReshapeAttrs
    adl_serializer<FlexFlow::ReshapeAttrs>::from_json(json const &j) {
  return {j.at("shape").template get<::FlexFlow::TensorShape>()};
}
void adl_serializer<FlexFlow::ReshapeAttrs>::to_json(
    json &j, FlexFlow::ReshapeAttrs const &v) {
  j["__type"] = "ReshapeAttrs";
  j["shape"] = v.shape;
}
} // namespace nlohmann

namespace rc {
Gen<FlexFlow::ReshapeAttrs> Arbitrary<FlexFlow::ReshapeAttrs>::arbitrary() {
  return gen::construct<FlexFlow::ReshapeAttrs>(
      gen::arbitrary<::FlexFlow::TensorShape>());
}
} // namespace rc

namespace FlexFlow {
std::string format_as(ReshapeAttrs const &x) {
  std::ostringstream oss;
  oss << "<ReshapeAttrs";
  oss << " shape=" << x.shape;
  oss << ">";
  return oss.str();
}
std::ostream &operator<<(std::ostream &s, ReshapeAttrs const &x) {
  return s << fmt::to_string(x);
}
} // namespace FlexFlow
