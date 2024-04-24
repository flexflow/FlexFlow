// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/pcg/include/pcg/initializers/glorot_uniform_attrs.struct.toml
/* proj-data
{
  "generated_from": "a268b411b6d378faa11e60c8517d7be5"
}
*/

#include "pcg/initializers/glorot_uniform_attrs.dtg.h"

#include <sstream>

namespace FlexFlow {
GlorotUniformAttrs::GlorotUniformAttrs(int const &seed) : seed(seed) {}
bool GlorotUniformAttrs::operator==(GlorotUniformAttrs const &other) const {
  return std::tie(this->seed) == std::tie(other.seed);
}
bool GlorotUniformAttrs::operator!=(GlorotUniformAttrs const &other) const {
  return std::tie(this->seed) != std::tie(other.seed);
}
bool GlorotUniformAttrs::operator<(GlorotUniformAttrs const &other) const {
  return std::tie(this->seed) < std::tie(other.seed);
}
bool GlorotUniformAttrs::operator>(GlorotUniformAttrs const &other) const {
  return std::tie(this->seed) > std::tie(other.seed);
}
bool GlorotUniformAttrs::operator<=(GlorotUniformAttrs const &other) const {
  return std::tie(this->seed) <= std::tie(other.seed);
}
bool GlorotUniformAttrs::operator>=(GlorotUniformAttrs const &other) const {
  return std::tie(this->seed) >= std::tie(other.seed);
}
} // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::GlorotUniformAttrs>::operator()(
    FlexFlow::GlorotUniformAttrs const &x) const {
  size_t result = 0;
  result ^=
      std::hash<int>{}(x.seed) + 0x9e3779b9 + (result << 6) + (result >> 2);
  return result;
}
} // namespace std

namespace nlohmann {
FlexFlow::GlorotUniformAttrs
    adl_serializer<FlexFlow::GlorotUniformAttrs>::from_json(json const &j) {
  return {j.at("seed").template get<int>()};
}
void adl_serializer<FlexFlow::GlorotUniformAttrs>::to_json(
    json &j, FlexFlow::GlorotUniformAttrs const &v) {
  j["__type"] = "GlorotUniformAttrs";
  j["seed"] = v.seed;
}
} // namespace nlohmann

namespace rc {
Gen<FlexFlow::GlorotUniformAttrs>
    Arbitrary<FlexFlow::GlorotUniformAttrs>::arbitrary() {
  return gen::construct<FlexFlow::GlorotUniformAttrs>(gen::arbitrary<int>());
}
} // namespace rc

namespace FlexFlow {
std::string format_as(GlorotUniformAttrs const &x) {
  std::ostringstream oss;
  oss << "<GlorotUniformAttrs";
  oss << " seed=" << x.seed;
  oss << ">";
  return oss.str();
}
std::ostream &operator<<(std::ostream &s, GlorotUniformAttrs const &x) {
  return s << fmt::to_string(x);
}
} // namespace FlexFlow
