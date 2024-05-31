// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/substitutions/include/substitutions/operator_pattern/operator_attribute_value.variant.toml
/* proj-data
{
  "generated_from": "de14592f1f4bcfb52689bc95e9d3b55f"
}
*/

#include "substitutions/operator_pattern/operator_attribute_value.dtg.h"

#include "fmt/format.h"
#include <sstream>
#include <stdexcept>

namespace FlexFlow {
OperatorAttributeValue::OperatorAttributeValue(int const &v) : raw_variant(v) {}
OperatorAttributeValue::OperatorAttributeValue(bool const &v)
    : raw_variant(v) {}
OperatorAttributeValue::OperatorAttributeValue(std::vector<int> const &v)
    : raw_variant(v) {}
OperatorAttributeValue::OperatorAttributeValue(
    std::vector<::FlexFlow::ff_dim_t> const &v)
    : raw_variant(v) {}
OperatorAttributeValue::OperatorAttributeValue(
    ::FlexFlow::OperatorType const &v)
    : raw_variant(v) {}
OperatorAttributeValue::OperatorAttributeValue(::FlexFlow::Activation const &v)
    : raw_variant(v) {}
OperatorAttributeValue::OperatorAttributeValue(::FlexFlow::ff_dim_t const &v)
    : raw_variant(v) {}
OperatorAttributeValue::OperatorAttributeValue(size_t const &v)
    : raw_variant(v) {}
OperatorAttributeValue::OperatorAttributeValue(::FlexFlow::AggregateOp const &v)
    : raw_variant(v) {}
OperatorAttributeValue::OperatorAttributeValue(
    std::optional<::FlexFlow::RegularizerAttrs> const &v)
    : raw_variant(v) {}
OperatorAttributeValue::OperatorAttributeValue(::FlexFlow::PoolOp const &v)
    : raw_variant(v) {}
OperatorAttributeValue::OperatorAttributeValue(::FlexFlow::TensorShape const &v)
    : raw_variant(v) {}
OperatorAttributeValue::OperatorAttributeValue(::FlexFlow::DataType const &v)
    : raw_variant(v) {}
bool OperatorAttributeValue::operator==(
    OperatorAttributeValue const &other) const {
  return this->raw_variant == other.raw_variant;
}
bool OperatorAttributeValue::operator!=(
    OperatorAttributeValue const &other) const {
  return this->raw_variant != other.raw_variant;
}
bool OperatorAttributeValue::operator<(
    OperatorAttributeValue const &other) const {
  return this->raw_variant < other.raw_variant;
}
bool OperatorAttributeValue::operator>(
    OperatorAttributeValue const &other) const {
  return this->raw_variant > other.raw_variant;
}
bool OperatorAttributeValue::operator<=(
    OperatorAttributeValue const &other) const {
  return this->raw_variant <= other.raw_variant;
}
bool OperatorAttributeValue::operator>=(
    OperatorAttributeValue const &other) const {
  return this->raw_variant >= other.raw_variant;
}
} // namespace FlexFlow
namespace std {
size_t hash<::FlexFlow::OperatorAttributeValue>::operator()(
    ::FlexFlow::OperatorAttributeValue const &x) const {
  return std::hash<std::variant<int,
                                bool,
                                std::vector<int>,
                                std::vector<::FlexFlow::ff_dim_t>,
                                ::FlexFlow::OperatorType,
                                ::FlexFlow::Activation,
                                ::FlexFlow::ff_dim_t,
                                size_t,
                                ::FlexFlow::AggregateOp,
                                std::optional<::FlexFlow::RegularizerAttrs>,
                                ::FlexFlow::PoolOp,
                                ::FlexFlow::TensorShape,
                                ::FlexFlow::DataType>>{}(x.raw_variant);
}
} // namespace std
namespace nlohmann {
::FlexFlow::OperatorAttributeValue
    adl_serializer<::FlexFlow::OperatorAttributeValue>::from_json(
        json const &j) {
  std::string key = j.at("type").template get<std::string>();
  if (key == "int") {
    return ::FlexFlow::OperatorAttributeValue{
        j.at("value").template get<int>()};
  } else if (key == "bool") {
    return ::FlexFlow::OperatorAttributeValue{
        j.at("value").template get<bool>()};
  } else if (key == "std::vector<int>") {
    return ::FlexFlow::OperatorAttributeValue{
        j.at("value").template get<std::vector<int>>()};
  } else if (key == "std::vector<::FlexFlow::ff_dim_t>") {
    return ::FlexFlow::OperatorAttributeValue{
        j.at("value").template get<std::vector<::FlexFlow::ff_dim_t>>()};
  } else if (key == "::FlexFlow::OperatorType") {
    return ::FlexFlow::OperatorAttributeValue{
        j.at("value").template get<::FlexFlow::OperatorType>()};
  } else if (key == "::FlexFlow::Activation") {
    return ::FlexFlow::OperatorAttributeValue{
        j.at("value").template get<::FlexFlow::Activation>()};
  } else if (key == "::FlexFlow::ff_dim_t") {
    return ::FlexFlow::OperatorAttributeValue{
        j.at("value").template get<::FlexFlow::ff_dim_t>()};
  } else if (key == "size_t") {
    return ::FlexFlow::OperatorAttributeValue{
        j.at("value").template get<size_t>()};
  } else if (key == "::FlexFlow::AggregateOp") {
    return ::FlexFlow::OperatorAttributeValue{
        j.at("value").template get<::FlexFlow::AggregateOp>()};
  } else if (key == "std::optional<::FlexFlow::RegularizerAttrs>") {
    return ::FlexFlow::OperatorAttributeValue{
        j.at("value")
            .template get<std::optional<::FlexFlow::RegularizerAttrs>>()};
  } else if (key == "::FlexFlow::PoolOp") {
    return ::FlexFlow::OperatorAttributeValue{
        j.at("value").template get<::FlexFlow::PoolOp>()};
  } else if (key == "::FlexFlow::TensorShape") {
    return ::FlexFlow::OperatorAttributeValue{
        j.at("value").template get<::FlexFlow::TensorShape>()};
  } else if (key == "::FlexFlow::DataType") {
    return ::FlexFlow::OperatorAttributeValue{
        j.at("value").template get<::FlexFlow::DataType>()};
  } else {
    throw std::runtime_error(fmt::format("Unknown type key {}", key));
  }
}
void adl_serializer<::FlexFlow::OperatorAttributeValue>::to_json(
    json &j, ::FlexFlow::OperatorAttributeValue const &x) {
  j["__type"] = "OperatorAttributeValue";
  switch (x.index()) {
    case 0: {
      j["type"] = "int";
      j["value"] = x.get<int>();
      break;
    }
    case 1: {
      j["type"] = "bool";
      j["value"] = x.get<bool>();
      break;
    }
    case 2: {
      j["type"] = "std::vector<int>";
      j["value"] = x.get<std::vector<int>>();
      break;
    }
    case 3: {
      j["type"] = "std::vector<::FlexFlow::ff_dim_t>";
      j["value"] = x.get<std::vector<::FlexFlow::ff_dim_t>>();
      break;
    }
    case 4: {
      j["type"] = "::FlexFlow::OperatorType";
      j["value"] = x.get<::FlexFlow::OperatorType>();
      break;
    }
    case 5: {
      j["type"] = "::FlexFlow::Activation";
      j["value"] = x.get<::FlexFlow::Activation>();
      break;
    }
    case 6: {
      j["type"] = "::FlexFlow::ff_dim_t";
      j["value"] = x.get<::FlexFlow::ff_dim_t>();
      break;
    }
    case 7: {
      j["type"] = "size_t";
      j["value"] = x.get<size_t>();
      break;
    }
    case 8: {
      j["type"] = "::FlexFlow::AggregateOp";
      j["value"] = x.get<::FlexFlow::AggregateOp>();
      break;
    }
    case 9: {
      j["type"] = "std::optional<::FlexFlow::RegularizerAttrs>";
      j["value"] = x.get<std::optional<::FlexFlow::RegularizerAttrs>>();
      break;
    }
    case 10: {
      j["type"] = "::FlexFlow::PoolOp";
      j["value"] = x.get<::FlexFlow::PoolOp>();
      break;
    }
    case 11: {
      j["type"] = "::FlexFlow::TensorShape";
      j["value"] = x.get<::FlexFlow::TensorShape>();
      break;
    }
    case 12: {
      j["type"] = "::FlexFlow::DataType";
      j["value"] = x.get<::FlexFlow::DataType>();
      break;
    }
    default: {
      throw std::runtime_error(fmt::format(
          "Unknown index {} for type OperatorAttributeValue", x.index()));
    }
  }
}
} // namespace nlohmann
namespace FlexFlow {
std::string format_as(::FlexFlow::OperatorAttributeValue const &x) {
  std::ostringstream oss;
  switch (x.index()) {
    case 0: {
      oss << "<OperatorAttributeValue int=" << x.get<int>() << ">";
      break;
    }
    case 1: {
      oss << "<OperatorAttributeValue bool=" << x.get<bool>() << ">";
      break;
    }
    case 2: {
      oss << "<OperatorAttributeValue std::vector<int>="
          << x.get<std::vector<int>>() << ">";
      break;
    }
    case 3: {
      oss << "<OperatorAttributeValue std::vector<::FlexFlow::ff_dim_t>="
          << x.get<std::vector<::FlexFlow::ff_dim_t>>() << ">";
      break;
    }
    case 4: {
      oss << "<OperatorAttributeValue ::FlexFlow::OperatorType="
          << x.get<::FlexFlow::OperatorType>() << ">";
      break;
    }
    case 5: {
      oss << "<OperatorAttributeValue ::FlexFlow::Activation="
          << x.get<::FlexFlow::Activation>() << ">";
      break;
    }
    case 6: {
      oss << "<OperatorAttributeValue ::FlexFlow::ff_dim_t="
          << x.get<::FlexFlow::ff_dim_t>() << ">";
      break;
    }
    case 7: {
      oss << "<OperatorAttributeValue size_t=" << x.get<size_t>() << ">";
      break;
    }
    case 8: {
      oss << "<OperatorAttributeValue ::FlexFlow::AggregateOp="
          << x.get<::FlexFlow::AggregateOp>() << ">";
      break;
    }
    case 9: {
      oss << "<OperatorAttributeValue "
             "std::optional<::FlexFlow::RegularizerAttrs>="
          << x.get<std::optional<::FlexFlow::RegularizerAttrs>>() << ">";
      break;
    }
    case 10: {
      oss << "<OperatorAttributeValue ::FlexFlow::PoolOp="
          << x.get<::FlexFlow::PoolOp>() << ">";
      break;
    }
    case 11: {
      oss << "<OperatorAttributeValue ::FlexFlow::TensorShape="
          << x.get<::FlexFlow::TensorShape>() << ">";
      break;
    }
    case 12: {
      oss << "<OperatorAttributeValue ::FlexFlow::DataType="
          << x.get<::FlexFlow::DataType>() << ">";
      break;
    }
    default: {
      throw std::runtime_error(fmt::format(
          "Unknown index {} for type OperatorAttributeValue", x.index()));
      break;
    }
  }
  return oss.str();
}
std::ostream &operator<<(std::ostream &s,
                         ::FlexFlow::OperatorAttributeValue const &x) {
  return s << fmt::to_string(x);
}
} // namespace FlexFlow