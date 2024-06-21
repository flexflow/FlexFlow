// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/utils/include/utils/graph/serial_parallel/split_type.enum.toml
/* proj-data
{
  "generated_from": "61d75c03b0273d05eb9707f75132974e"
}
*/

#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_SPLIT_TYPE_DTG_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_SPLIT_TYPE_DTG_H

#include "fmt/format.h"
#include "nlohmann/json.hpp"
#include "rapidcheck.h"
#include <functional>
#include <ostream>
#include <string>

namespace FlexFlow {
enum class SplitType { SERIAL, PARALLEL };
std::string format_as(SplitType);
std::ostream &operator<<(std::ostream &, SplitType);
void to_json(::nlohmann::json &, SplitType);
void from_json(::nlohmann::json const &, SplitType &);
} // namespace FlexFlow
namespace std {
template <>
struct hash<FlexFlow::SplitType> {
  size_t operator()(FlexFlow::SplitType) const;
};
} // namespace std
namespace rc {
template <>
struct Arbitrary<FlexFlow::SplitType> {
  static Gen<FlexFlow::SplitType> arbitrary();
};
} // namespace rc

#endif // _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_SPLIT_TYPE_DTG_H
