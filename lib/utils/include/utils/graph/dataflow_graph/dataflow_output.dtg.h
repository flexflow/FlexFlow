// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/utils/include/utils/graph/dataflow_graph/dataflow_output.struct.toml
/* proj-data
{
  "generated_from": "3f4ea6635782f141cc593291132c4064"
}
*/

#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_DATAFLOW_OUTPUT_DTG_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_DATAFLOW_OUTPUT_DTG_H

#include "fmt/format.h"
#include "utils/graph/node/node.dtg.h"
#include <functional>
#include <ostream>
#include <tuple>

namespace FlexFlow {
struct DataflowOutput {
  DataflowOutput() = delete;
  explicit DataflowOutput(::FlexFlow::Node const &node, int const &idx);

  bool operator==(DataflowOutput const &) const;
  bool operator!=(DataflowOutput const &) const;
  bool operator<(DataflowOutput const &) const;
  bool operator>(DataflowOutput const &) const;
  bool operator<=(DataflowOutput const &) const;
  bool operator>=(DataflowOutput const &) const;
  ::FlexFlow::Node node;
  int idx;
};
} // namespace FlexFlow

namespace std {
template <>
struct hash<::FlexFlow::DataflowOutput> {
  size_t operator()(::FlexFlow::DataflowOutput const &) const;
};
} // namespace std

namespace FlexFlow {
std::string format_as(DataflowOutput const &);
std::ostream &operator<<(std::ostream &, DataflowOutput const &);
} // namespace FlexFlow

#endif // _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_DATAFLOW_OUTPUT_DTG_H
