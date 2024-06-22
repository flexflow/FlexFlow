// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/op-attrs/include/op-attrs/ops/repartition_attrs.struct.toml
/* proj-data
{
  "generated_from": "0a4d8b435768ce3ee37013fc550c9ebb"
}
*/

#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_REPARTITION_ATTRS_DTG_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_REPARTITION_ATTRS_DTG_H

#include "fmt/format.h"
#include "nlohmann/json.hpp"
#include "op-attrs/ff_dim.dtg.h"
#include "op-attrs/ff_dim.h"
#include "rapidcheck.h"
#include <functional>
#include <ostream>
#include <tuple>

namespace FlexFlow {
struct RepartitionAttrs {
  RepartitionAttrs() = delete;
  explicit RepartitionAttrs(::FlexFlow::ff_dim_t const &repartition_dim,
                            int const &repartition_degree);

  bool operator==(RepartitionAttrs const &) const;
  bool operator!=(RepartitionAttrs const &) const;
  bool operator<(RepartitionAttrs const &) const;
  bool operator>(RepartitionAttrs const &) const;
  bool operator<=(RepartitionAttrs const &) const;
  bool operator>=(RepartitionAttrs const &) const;
  ::FlexFlow::ff_dim_t repartition_dim;
  int repartition_degree;
};
} // namespace FlexFlow

namespace std {
template <>
struct hash<::FlexFlow::RepartitionAttrs> {
  size_t operator()(::FlexFlow::RepartitionAttrs const &) const;
};
} // namespace std

namespace nlohmann {
template <>
struct adl_serializer<::FlexFlow::RepartitionAttrs> {
  static ::FlexFlow::RepartitionAttrs from_json(json const &);
  static void to_json(json &, ::FlexFlow::RepartitionAttrs const &);
};
} // namespace nlohmann

namespace rc {
template <>
struct Arbitrary<::FlexFlow::RepartitionAttrs> {
  static Gen<::FlexFlow::RepartitionAttrs> arbitrary();
};
} // namespace rc

namespace FlexFlow {
std::string format_as(RepartitionAttrs const &);
std::ostream &operator<<(std::ostream &, RepartitionAttrs const &);
} // namespace FlexFlow

#endif // _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_REPARTITION_ATTRS_DTG_H
