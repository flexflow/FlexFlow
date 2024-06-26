// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/pcg/include/pcg/layer_guid_t.struct.toml
/* proj-data
{
  "generated_from": "a672ffe470fd1dde8299f91f3038ca7a"
}
*/

#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_LAYER_GUID_T_DTG_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_LAYER_GUID_T_DTG_H

#include "fmt/format.h"
#include "utils/graph.h"
#include <functional>
#include <ostream>
#include <tuple>

namespace FlexFlow {
struct layer_guid_t {
  layer_guid_t() = delete;
  explicit layer_guid_t(::FlexFlow::Node const &raw_node);

  bool operator==(layer_guid_t const &) const;
  bool operator!=(layer_guid_t const &) const;
  bool operator<(layer_guid_t const &) const;
  bool operator>(layer_guid_t const &) const;
  bool operator<=(layer_guid_t const &) const;
  bool operator>=(layer_guid_t const &) const;
  ::FlexFlow::Node raw_node;
};
} // namespace FlexFlow

namespace std {
template <>
struct hash<::FlexFlow::layer_guid_t> {
  size_t operator()(::FlexFlow::layer_guid_t const &) const;
};
} // namespace std

namespace FlexFlow {
std::string format_as(layer_guid_t const &);
std::ostream &operator<<(std::ostream &, layer_guid_t const &);
} // namespace FlexFlow

#endif // _FLEXFLOW_LIB_PCG_INCLUDE_PCG_LAYER_GUID_T_DTG_H
