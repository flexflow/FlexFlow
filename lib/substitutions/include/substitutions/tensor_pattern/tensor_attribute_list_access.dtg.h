// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/substitutions/include/substitutions/tensor_pattern/tensor_attribute_list_access.struct.toml
/* proj-data
{
  "generated_from": "41f5449cd700b6d7ab017f3efa39dc1d"
}
*/

#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_TENSOR_PATTERN_TENSOR_ATTRIBUTE_LIST_ACCESS_DTG_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_TENSOR_PATTERN_TENSOR_ATTRIBUTE_LIST_ACCESS_DTG_H

#include "fmt/format.h"
#include "nlohmann/json.hpp"
#include "rapidcheck.h"
#include "substitutions/tensor_pattern/tensor_attribute_key.dtg.h"
#include <functional>
#include <ostream>
#include <tuple>

namespace FlexFlow {
struct TensorAttributeListIndexAccess {
  TensorAttributeListIndexAccess() = delete;
  TensorAttributeListIndexAccess(
      ::FlexFlow::TensorAttributeKey const &attribute_key, int const &index);

  bool operator==(TensorAttributeListIndexAccess const &) const;
  bool operator!=(TensorAttributeListIndexAccess const &) const;
  bool operator<(TensorAttributeListIndexAccess const &) const;
  bool operator>(TensorAttributeListIndexAccess const &) const;
  bool operator<=(TensorAttributeListIndexAccess const &) const;
  bool operator>=(TensorAttributeListIndexAccess const &) const;
  ::FlexFlow::TensorAttributeKey attribute_key;
  int index;
};
} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::TensorAttributeListIndexAccess> {
  size_t operator()(FlexFlow::TensorAttributeListIndexAccess const &) const;
};
} // namespace std

namespace nlohmann {
template <>
struct adl_serializer<FlexFlow::TensorAttributeListIndexAccess> {
  static FlexFlow::TensorAttributeListIndexAccess from_json(json const &);
  static void to_json(json &, FlexFlow::TensorAttributeListIndexAccess const &);
};
} // namespace nlohmann

namespace rc {
template <>
struct Arbitrary<FlexFlow::TensorAttributeListIndexAccess> {
  static Gen<FlexFlow::TensorAttributeListIndexAccess> arbitrary();
};
} // namespace rc

namespace FlexFlow {
std::string format_as(TensorAttributeListIndexAccess const &);
std::ostream &operator<<(std::ostream &,
                         TensorAttributeListIndexAccess const &);
} // namespace FlexFlow

#endif // _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_TENSOR_PATTERN_TENSOR_ATTRIBUTE_LIST_ACCESS_DTG_H
