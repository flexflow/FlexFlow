// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/pcg/include/pcg/initializers/norm_initializer_attrs.struct.toml
/* proj-data
{
  "generated_from": "6843fc9ca02aea2b40e57dbc497f99ac"
}
*/

#include "pcg/initializers/norm_initializer_attrs.dtg.h"

#include <sstream>

namespace FlexFlow {
NormInitializerAttrs::NormInitializerAttrs(int const &seed,
                                           float const &mean,
                                           float const &stddev)
    : seed(seed), mean(mean), stddev(stddev) {}
bool NormInitializerAttrs::operator==(NormInitializerAttrs const &other) const {
  return std::tie(this->seed, this->mean, this->stddev) ==
         std::tie(other.seed, other.mean, other.stddev);
}
bool NormInitializerAttrs::operator!=(NormInitializerAttrs const &other) const {
  return std::tie(this->seed, this->mean, this->stddev) !=
         std::tie(other.seed, other.mean, other.stddev);
}
bool NormInitializerAttrs::operator<(NormInitializerAttrs const &other) const {
  return std::tie(this->seed, this->mean, this->stddev) <
         std::tie(other.seed, other.mean, other.stddev);
}
bool NormInitializerAttrs::operator>(NormInitializerAttrs const &other) const {
  return std::tie(this->seed, this->mean, this->stddev) >
         std::tie(other.seed, other.mean, other.stddev);
}
bool NormInitializerAttrs::operator<=(NormInitializerAttrs const &other) const {
  return std::tie(this->seed, this->mean, this->stddev) <=
         std::tie(other.seed, other.mean, other.stddev);
}
bool NormInitializerAttrs::operator>=(NormInitializerAttrs const &other) const {
  return std::tie(this->seed, this->mean, this->stddev) >=
         std::tie(other.seed, other.mean, other.stddev);
}
} // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::NormInitializerAttrs>::operator()(
    ::FlexFlow::NormInitializerAttrs const &x) const {
  size_t result = 0;
  result ^=
      std::hash<int>{}(x.seed) + 0x9e3779b9 + (result << 6) + (result >> 2);
  result ^=
      std::hash<float>{}(x.mean) + 0x9e3779b9 + (result << 6) + (result >> 2);
  result ^=
      std::hash<float>{}(x.stddev) + 0x9e3779b9 + (result << 6) + (result >> 2);
  return result;
}
} // namespace std

namespace nlohmann {
::FlexFlow::NormInitializerAttrs
    adl_serializer<::FlexFlow::NormInitializerAttrs>::from_json(json const &j) {
  return ::FlexFlow::NormInitializerAttrs{j.at("seed").template get<int>(),
                                          j.at("mean").template get<float>(),
                                          j.at("stddev").template get<float>()};
}
void adl_serializer<::FlexFlow::NormInitializerAttrs>::to_json(
    json &j, ::FlexFlow::NormInitializerAttrs const &v) {
  j["__type"] = "NormInitializerAttrs";
  j["seed"] = v.seed;
  j["mean"] = v.mean;
  j["stddev"] = v.stddev;
}
} // namespace nlohmann

namespace rc {
Gen<::FlexFlow::NormInitializerAttrs>
    Arbitrary<::FlexFlow::NormInitializerAttrs>::arbitrary() {
  return gen::construct<::FlexFlow::NormInitializerAttrs>(
      gen::arbitrary<int>(), gen::arbitrary<float>(), gen::arbitrary<float>());
}
} // namespace rc

namespace FlexFlow {
std::string format_as(NormInitializerAttrs const &x) {
  std::ostringstream oss;
  oss << "<NormInitializerAttrs";
  oss << " seed=" << x.seed;
  oss << " mean=" << x.mean;
  oss << " stddev=" << x.stddev;
  oss << ">";
  return oss.str();
}
std::ostream &operator<<(std::ostream &s, NormInitializerAttrs const &x) {
  return s << fmt::to_string(x);
}
} // namespace FlexFlow