// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/op-attrs/include/op-attrs/ops/embedding_attrs.struct.toml

#include "op-attrs/ops/embedding_attrs.h"

namespace FlexFlow {
EmbeddingAttrs::EmbeddingAttrs(int const &num_entries,
                               int const &out_channels,
                               ::FlexFlow::AggregateOp const &aggr,
                               ::FlexFlow::DataType const &data_type)
    : num_entries(num_entries), out_channels(out_channels), aggr(aggr),
      data_type(data_type) {}
bool EmbeddingAttrs::operator==(EmbeddingAttrs const &other) const {
  return std::tie(this->num_entries,
                  this->out_channels,
                  this->aggr,
                  this->data_type) == std::tie(other.num_entries,
                                               other.out_channels,
                                               other.aggr,
                                               other.data_type);
}
bool EmbeddingAttrs::operator!=(EmbeddingAttrs const &other) const {
  return std::tie(this->num_entries,
                  this->out_channels,
                  this->aggr,
                  this->data_type) != std::tie(other.num_entries,
                                               other.out_channels,
                                               other.aggr,
                                               other.data_type);
}
bool EmbeddingAttrs::operator<(EmbeddingAttrs const &other) const {
  return std::tie(this->num_entries,
                  this->out_channels,
                  this->aggr,
                  this->data_type) < std::tie(other.num_entries,
                                              other.out_channels,
                                              other.aggr,
                                              other.data_type);
}
bool EmbeddingAttrs::operator>(EmbeddingAttrs const &other) const {
  return std::tie(this->num_entries,
                  this->out_channels,
                  this->aggr,
                  this->data_type) > std::tie(other.num_entries,
                                              other.out_channels,
                                              other.aggr,
                                              other.data_type);
}
bool EmbeddingAttrs::operator<=(EmbeddingAttrs const &other) const {
  return std::tie(this->num_entries,
                  this->out_channels,
                  this->aggr,
                  this->data_type) <= std::tie(other.num_entries,
                                               other.out_channels,
                                               other.aggr,
                                               other.data_type);
}
bool EmbeddingAttrs::operator>=(EmbeddingAttrs const &other) const {
  return std::tie(this->num_entries,
                  this->out_channels,
                  this->aggr,
                  this->data_type) >= std::tie(other.num_entries,
                                               other.out_channels,
                                               other.aggr,
                                               other.data_type);
}
} // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::EmbeddingAttrs>::operator()(
    FlexFlow::EmbeddingAttrs const &x) const {
  size_t result = 0;
  result ^= std::hash<int>{}(x.num_entries) + 0x9e3779b9 + (result << 6) +
            (result >> 2);
  result ^= std::hash<int>{}(x.out_channels) + 0x9e3779b9 + (result << 6) +
            (result >> 2);
  result ^= std::hash<::FlexFlow::AggregateOp>{}(x.aggr) + 0x9e3779b9 +
            (result << 6) + (result >> 2);
  result ^= std::hash<::FlexFlow::DataType>{}(x.data_type) + 0x9e3779b9 +
            (result << 6) + (result >> 2);
  return result;
}
} // namespace std

namespace nlohmann {
FlexFlow::EmbeddingAttrs
    adl_serializer<FlexFlow::EmbeddingAttrs>::from_json(json const &j) {
  return {j.at("num_entries").template get<int>(),
          j.at("out_channels").template get<int>(),
          j.at("aggr").template get<::FlexFlow::AggregateOp>(),
          j.at("data_type").template get<::FlexFlow::DataType>()};
}
void adl_serializer<FlexFlow::EmbeddingAttrs>::to_json(
    json &j, FlexFlow::EmbeddingAttrs const &v) {
  j["__type"] = "EmbeddingAttrs";
  j["num_entries"] = v.num_entries;
  j["out_channels"] = v.out_channels;
  j["aggr"] = v.aggr;
  j["data_type"] = v.data_type;
}
} // namespace nlohmann

namespace rc {
Gen<FlexFlow::EmbeddingAttrs> Arbitrary<FlexFlow::EmbeddingAttrs>::arbitrary() {
  return gen::construct<FlexFlow::EmbeddingAttrs>(
      gen::arbitrary<int>(),
      gen::arbitrary<int>(),
      gen::arbitrary<::FlexFlow::AggregateOp>(),
      gen::arbitrary<::FlexFlow::DataType>());
}
} // namespace rc

namespace FlexFlow {
std::string format_as(EmbeddingAttrs const &x) {
  std::ostringstream oss;
  oss << "<EmbeddingAttrs";
  oss << " num_entries=" << x.num_entries;
  oss << " out_channels=" << x.out_channels;
  oss << " aggr=" << x.aggr;
  oss << " data_type=" << x.data_type;
  oss << ">";
  return oss.str();
}
std::ostream &operator<<(std::ostream &s, EmbeddingAttrs const &x) {
  return s << fmt::to_string(x);
}
} // namespace FlexFlow
