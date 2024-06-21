// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/utils/include/utils/graph/dataflow_graph/dataflow_output.struct.toml
/* proj-data
{
  "generated_from": "3f4ea6635782f141cc593291132c4064"
}
*/

#include "utils/graph/dataflow_graph/dataflow_output.dtg.h"

#include <sstream>

namespace FlexFlow {
DataflowOutput::DataflowOutput(::FlexFlow::Node const &node, int const &idx)
    : node(node), idx(idx) {}
bool DataflowOutput::operator==(DataflowOutput const &other) const {
  return std::tie(this->node, this->idx) == std::tie(other.node, other.idx);
}
bool DataflowOutput::operator!=(DataflowOutput const &other) const {
  return std::tie(this->node, this->idx) != std::tie(other.node, other.idx);
}
bool DataflowOutput::operator<(DataflowOutput const &other) const {
  return std::tie(this->node, this->idx) < std::tie(other.node, other.idx);
}
bool DataflowOutput::operator>(DataflowOutput const &other) const {
  return std::tie(this->node, this->idx) > std::tie(other.node, other.idx);
}
bool DataflowOutput::operator<=(DataflowOutput const &other) const {
  return std::tie(this->node, this->idx) <= std::tie(other.node, other.idx);
}
bool DataflowOutput::operator>=(DataflowOutput const &other) const {
  return std::tie(this->node, this->idx) >= std::tie(other.node, other.idx);
}
} // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::DataflowOutput>::operator()(
    ::FlexFlow::DataflowOutput const &x) const {
  size_t result = 0;
  result ^= std::hash<::FlexFlow::Node>{}(x.node) + 0x9e3779b9 + (result << 6) +
            (result >> 2);
  result ^=
      std::hash<int>{}(x.idx) + 0x9e3779b9 + (result << 6) + (result >> 2);
  return result;
}
} // namespace std

namespace FlexFlow {
std::string format_as(DataflowOutput const &x) {
  std::ostringstream oss;
  oss << "<DataflowOutput";
  oss << " node=" << x.node;
  oss << " idx=" << x.idx;
  oss << ">";
  return oss.str();
}
std::ostream &operator<<(std::ostream &s, DataflowOutput const &x) {
  return s << fmt::to_string(x);
}
} // namespace FlexFlow
