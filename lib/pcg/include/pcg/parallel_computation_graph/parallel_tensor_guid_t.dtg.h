// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/pcg/include/pcg/parallel_computation_graph/parallel_tensor_guid_t.struct.toml
/* proj-data
{
  "generated_from": "ff4f90460638385dc94c7f0e87a0bf7f"
}
*/

#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_PARALLEL_COMPUTATION_GRAPH_PARALLEL_TENSOR_GUID_T_DTG_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_PARALLEL_COMPUTATION_GRAPH_PARALLEL_TENSOR_GUID_T_DTG_H

#include "fmt/format.h"
#include "utils/graph/dataflow_graph/dataflow_output.dtg.h"
#include <functional>
#include <ostream>
#include <tuple>

namespace FlexFlow {
struct parallel_tensor_guid_t {
  parallel_tensor_guid_t() = delete;
  explicit parallel_tensor_guid_t(
      ::FlexFlow::DataflowOutput const &raw_graph_output);

  bool operator==(parallel_tensor_guid_t const &) const;
  bool operator!=(parallel_tensor_guid_t const &) const;
  bool operator<(parallel_tensor_guid_t const &) const;
  bool operator>(parallel_tensor_guid_t const &) const;
  bool operator<=(parallel_tensor_guid_t const &) const;
  bool operator>=(parallel_tensor_guid_t const &) const;
  ::FlexFlow::DataflowOutput raw_graph_output;
};
} // namespace FlexFlow

namespace std {
template <>
struct hash<::FlexFlow::parallel_tensor_guid_t> {
  size_t operator()(::FlexFlow::parallel_tensor_guid_t const &) const;
};
} // namespace std

namespace FlexFlow {
std::string format_as(parallel_tensor_guid_t const &);
std::ostream &operator<<(std::ostream &, parallel_tensor_guid_t const &);
} // namespace FlexFlow

#endif // _FLEXFLOW_LIB_PCG_INCLUDE_PCG_PARALLEL_COMPUTATION_GRAPH_PARALLEL_TENSOR_GUID_T_DTG_H
