// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/op-attrs/include/op-attrs/tensor_shape.struct.toml
/* proj-data
{
  "generated_from": "ef6fa5088b89d6da4dc8bddf0a6d3294"
}
*/

#include "op-attrs/tensor_shape.dtg.h"

#include <sstream>

namespace FlexFlow {
TensorShape::TensorShape(::FlexFlow::TensorDims const &dims,
                         ::FlexFlow::DataType const &data_type)
    : dims(dims), data_type(data_type) {}
bool TensorShape::operator==(TensorShape const &other) const {
  return std::tie(this->dims, this->data_type) ==
         std::tie(other.dims, other.data_type);
}
bool TensorShape::operator!=(TensorShape const &other) const {
  return std::tie(this->dims, this->data_type) !=
         std::tie(other.dims, other.data_type);
}
bool TensorShape::operator<(TensorShape const &other) const {
  return std::tie(this->dims, this->data_type) <
         std::tie(other.dims, other.data_type);
}
bool TensorShape::operator>(TensorShape const &other) const {
  return std::tie(this->dims, this->data_type) >
         std::tie(other.dims, other.data_type);
}
bool TensorShape::operator<=(TensorShape const &other) const {
  return std::tie(this->dims, this->data_type) <=
         std::tie(other.dims, other.data_type);
}
bool TensorShape::operator>=(TensorShape const &other) const {
  return std::tie(this->dims, this->data_type) >=
         std::tie(other.dims, other.data_type);
}
} // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::TensorShape>::operator()(
    ::FlexFlow::TensorShape const &x) const {
  size_t result = 0;
  result ^= std::hash<::FlexFlow::TensorDims>{}(x.dims) + 0x9e3779b9 +
            (result << 6) + (result >> 2);
  result ^= std::hash<::FlexFlow::DataType>{}(x.data_type) + 0x9e3779b9 +
            (result << 6) + (result >> 2);
  return result;
}
} // namespace std

namespace nlohmann {
::FlexFlow::TensorShape
    adl_serializer<::FlexFlow::TensorShape>::from_json(json const &j) {
  return ::FlexFlow::TensorShape{
      j.at("dims").template get<::FlexFlow::TensorDims>(),
      j.at("data_type").template get<::FlexFlow::DataType>()};
}
void adl_serializer<::FlexFlow::TensorShape>::to_json(
    json &j, ::FlexFlow::TensorShape const &v) {
  j["__type"] = "TensorShape";
  j["dims"] = v.dims;
  j["data_type"] = v.data_type;
}
} // namespace nlohmann

namespace rc {
Gen<::FlexFlow::TensorShape> Arbitrary<::FlexFlow::TensorShape>::arbitrary() {
  return gen::construct<::FlexFlow::TensorShape>(
      gen::arbitrary<::FlexFlow::TensorDims>(),
      gen::arbitrary<::FlexFlow::DataType>());
}
} // namespace rc

namespace FlexFlow {
std::string format_as(TensorShape const &x) {
  std::ostringstream oss;
  oss << "<TensorShape";
  oss << " dims=" << x.dims;
  oss << " data_type=" << x.data_type;
  oss << ">";
  return oss.str();
}
std::ostream &operator<<(std::ostream &s, TensorShape const &x) {
  return s << fmt::to_string(x);
}
} // namespace FlexFlow
