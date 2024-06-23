// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/substitutions/include/substitutions/tensor_pattern/tensor_attribute_value.variant.toml
/* proj-data
{
  "generated_from": "c220bfd8b5a57e4941e4739c84d20054"
}
*/

#include "substitutions/tensor_pattern/tensor_attribute_value.dtg.h"

#include "fmt/format.h"
#include <sstream>
#include <stdexcept>

namespace FlexFlow {
TensorAttributeValue::TensorAttributeValue(size_t const &v) : raw_variant(v) {}
TensorAttributeValue::TensorAttributeValue(std::vector<size_t> const &v)
    : raw_variant(v) {}
bool TensorAttributeValue::operator==(TensorAttributeValue const &other) const {
  return this->raw_variant == other.raw_variant;
}
bool TensorAttributeValue::operator!=(TensorAttributeValue const &other) const {
  return this->raw_variant != other.raw_variant;
}
bool TensorAttributeValue::operator<(TensorAttributeValue const &other) const {
  return this->raw_variant < other.raw_variant;
}
bool TensorAttributeValue::operator>(TensorAttributeValue const &other) const {
  return this->raw_variant > other.raw_variant;
}
bool TensorAttributeValue::operator<=(TensorAttributeValue const &other) const {
  return this->raw_variant <= other.raw_variant;
}
bool TensorAttributeValue::operator>=(TensorAttributeValue const &other) const {
  return this->raw_variant >= other.raw_variant;
}
} // namespace FlexFlow
namespace std {
size_t hash<::FlexFlow::TensorAttributeValue>::operator()(
    ::FlexFlow::TensorAttributeValue const &x) const {
  return std::hash<std::variant<size_t, std::vector<size_t>>>{}(x.raw_variant);
}
} // namespace std
namespace nlohmann {
::FlexFlow::TensorAttributeValue
    adl_serializer<::FlexFlow::TensorAttributeValue>::from_json(json const &j) {
  std::string key = j.at("type").template get<std::string>();
  if (key == "size_t") {
    return ::FlexFlow::TensorAttributeValue{
        j.at("value").template get<size_t>()};
  } else if (key == "std::vector<size_t>") {
    return ::FlexFlow::TensorAttributeValue{
        j.at("value").template get<std::vector<size_t>>()};
  } else {
    throw std::runtime_error(fmt::format("Unknown type key {}", key));
  }
}
void adl_serializer<::FlexFlow::TensorAttributeValue>::to_json(
    json &j, ::FlexFlow::TensorAttributeValue const &x) {
  j["__type"] = "TensorAttributeValue";
  switch (x.index()) {
    case 0: {
      j["type"] = "size_t";
      j["value"] = x.get<size_t>();
      break;
    }
    case 1: {
      j["type"] = "std::vector<size_t>";
      j["value"] = x.get<std::vector<size_t>>();
      break;
    }
    default: {
      throw std::runtime_error(fmt::format(
          "Unknown index {} for type TensorAttributeValue", x.index()));
    }
  }
}
} // namespace nlohmann
namespace FlexFlow {
std::string format_as(::FlexFlow::TensorAttributeValue const &x) {
  std::ostringstream oss;
  switch (x.index()) {
    case 0: {
      oss << "<TensorAttributeValue size_t=" << x.get<size_t>() << ">";
      break;
    }
    case 1: {
      oss << "<TensorAttributeValue std::vector<size_t>="
          << x.get<std::vector<size_t>>() << ">";
      break;
    }
    default: {
      throw std::runtime_error(fmt::format(
          "Unknown index {} for type TensorAttributeValue", x.index()));
      break;
    }
  }
  return oss.str();
}
std::ostream &operator<<(std::ostream &s,
                         ::FlexFlow::TensorAttributeValue const &x) {
  return s << fmt::to_string(x);
}
} // namespace FlexFlow
