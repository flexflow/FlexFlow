// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/op-attrs/include/op-attrs/ops/layer_norm_attrs.struct.toml
/* proj-data
{
  "generated_from": "349deae8d9356d3eeacd7e7d069c3155"
}
*/

#include "op-attrs/ops/layer_norm_attrs.dtg.h"

#include <sstream>

namespace FlexFlow {
LayerNormAttrs::LayerNormAttrs(
    ::FlexFlow::stack_vector<::FlexFlow::ff_dim_t, MAX_TENSOR_DIM> const &axes,
    bool const &elementwise_affine,
    float const &eps)
    : axes(axes), elementwise_affine(elementwise_affine), eps(eps) {}
bool LayerNormAttrs::operator==(LayerNormAttrs const &other) const {
  return std::tie(this->axes, this->elementwise_affine, this->eps) ==
         std::tie(other.axes, other.elementwise_affine, other.eps);
}
bool LayerNormAttrs::operator!=(LayerNormAttrs const &other) const {
  return std::tie(this->axes, this->elementwise_affine, this->eps) !=
         std::tie(other.axes, other.elementwise_affine, other.eps);
}
bool LayerNormAttrs::operator<(LayerNormAttrs const &other) const {
  return std::tie(this->axes, this->elementwise_affine, this->eps) <
         std::tie(other.axes, other.elementwise_affine, other.eps);
}
bool LayerNormAttrs::operator>(LayerNormAttrs const &other) const {
  return std::tie(this->axes, this->elementwise_affine, this->eps) >
         std::tie(other.axes, other.elementwise_affine, other.eps);
}
bool LayerNormAttrs::operator<=(LayerNormAttrs const &other) const {
  return std::tie(this->axes, this->elementwise_affine, this->eps) <=
         std::tie(other.axes, other.elementwise_affine, other.eps);
}
bool LayerNormAttrs::operator>=(LayerNormAttrs const &other) const {
  return std::tie(this->axes, this->elementwise_affine, this->eps) >=
         std::tie(other.axes, other.elementwise_affine, other.eps);
}
} // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::LayerNormAttrs>::operator()(
    ::FlexFlow::LayerNormAttrs const &x) const {
  size_t result = 0;
  result ^=
      std::hash<
          ::FlexFlow::stack_vector<::FlexFlow::ff_dim_t, MAX_TENSOR_DIM>>{}(
          x.axes) +
      0x9e3779b9 + (result << 6) + (result >> 2);
  result ^= std::hash<bool>{}(x.elementwise_affine) + 0x9e3779b9 +
            (result << 6) + (result >> 2);
  result ^=
      std::hash<float>{}(x.eps) + 0x9e3779b9 + (result << 6) + (result >> 2);
  return result;
}
} // namespace std

namespace nlohmann {
::FlexFlow::LayerNormAttrs
    adl_serializer<::FlexFlow::LayerNormAttrs>::from_json(json const &j) {
  return ::FlexFlow::LayerNormAttrs{
      j.at("axes")
          .template get<
              ::FlexFlow::stack_vector<::FlexFlow::ff_dim_t, MAX_TENSOR_DIM>>(),
      j.at("elementwise_affine").template get<bool>(),
      j.at("eps").template get<float>()};
}
void adl_serializer<::FlexFlow::LayerNormAttrs>::to_json(
    json &j, ::FlexFlow::LayerNormAttrs const &v) {
  j["__type"] = "LayerNormAttrs";
  j["axes"] = v.axes;
  j["elementwise_affine"] = v.elementwise_affine;
  j["eps"] = v.eps;
}
} // namespace nlohmann

namespace rc {
Gen<::FlexFlow::LayerNormAttrs>
    Arbitrary<::FlexFlow::LayerNormAttrs>::arbitrary() {
  return gen::construct<::FlexFlow::LayerNormAttrs>(
      gen::arbitrary<
          ::FlexFlow::stack_vector<::FlexFlow::ff_dim_t, MAX_TENSOR_DIM>>(),
      gen::arbitrary<bool>(),
      gen::arbitrary<float>());
}
} // namespace rc

namespace FlexFlow {
std::string format_as(LayerNormAttrs const &x) {
  std::ostringstream oss;
  oss << "<LayerNormAttrs";
  oss << " axes=" << x.axes;
  oss << " elementwise_affine=" << x.elementwise_affine;
  oss << " eps=" << x.eps;
  oss << ">";
  return oss.str();
}
std::ostream &operator<<(std::ostream &s, LayerNormAttrs const &x) {
  return s << fmt::to_string(x);
}
} // namespace FlexFlow
