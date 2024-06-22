// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/pcg/include/pcg/side_size_t.struct.toml
/* proj-data
{
  "generated_from": "6a1669890e547dcc7a4ddb90be05be15"
}
*/

#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_SIDE_SIZE_T_DTG_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_SIDE_SIZE_T_DTG_H

#include "fmt/format.h"
#include "nlohmann/json.hpp"
#include "rapidcheck.h"
#include <functional>
#include <ostream>
#include <tuple>

namespace FlexFlow {
struct side_size_t {
  side_size_t() = delete;
  explicit side_size_t(int const &unwrapped);

  bool operator==(side_size_t const &) const;
  bool operator!=(side_size_t const &) const;
  bool operator<(side_size_t const &) const;
  bool operator>(side_size_t const &) const;
  bool operator<=(side_size_t const &) const;
  bool operator>=(side_size_t const &) const;
  int unwrapped;
};
} // namespace FlexFlow

namespace std {
template <>
struct hash<::FlexFlow::side_size_t> {
  size_t operator()(::FlexFlow::side_size_t const &) const;
};
} // namespace std

namespace nlohmann {
template <>
struct adl_serializer<::FlexFlow::side_size_t> {
  static ::FlexFlow::side_size_t from_json(json const &);
  static void to_json(json &, ::FlexFlow::side_size_t const &);
};
} // namespace nlohmann

namespace rc {
template <>
struct Arbitrary<::FlexFlow::side_size_t> {
  static Gen<::FlexFlow::side_size_t> arbitrary();
};
} // namespace rc

namespace FlexFlow {
std::string format_as(side_size_t const &);
std::ostream &operator<<(std::ostream &, side_size_t const &);
} // namespace FlexFlow

#endif // _FLEXFLOW_LIB_PCG_INCLUDE_PCG_SIDE_SIZE_T_DTG_H
