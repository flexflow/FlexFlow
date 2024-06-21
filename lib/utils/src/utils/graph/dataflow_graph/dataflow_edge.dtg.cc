// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/utils/include/utils/graph/dataflow_graph/dataflow_edge.struct.toml
/* proj-data
{
  "generated_from": "4728f139efc6884057f39e38f44a791b"
}
*/

#include "utils/graph/dataflow_graph/dataflow_edge.dtg.h"

#include <sstream>

namespace FlexFlow {
DataflowEdge::DataflowEdge(::FlexFlow::DataflowOutput const &src,
                           ::FlexFlow::DataflowInput const &dst)
    : src(src), dst(dst) {}
bool DataflowEdge::operator==(DataflowEdge const &other) const {
  return std::tie(this->src, this->dst) == std::tie(other.src, other.dst);
}
bool DataflowEdge::operator!=(DataflowEdge const &other) const {
  return std::tie(this->src, this->dst) != std::tie(other.src, other.dst);
}
bool DataflowEdge::operator<(DataflowEdge const &other) const {
  return std::tie(this->src, this->dst) < std::tie(other.src, other.dst);
}
bool DataflowEdge::operator>(DataflowEdge const &other) const {
  return std::tie(this->src, this->dst) > std::tie(other.src, other.dst);
}
bool DataflowEdge::operator<=(DataflowEdge const &other) const {
  return std::tie(this->src, this->dst) <= std::tie(other.src, other.dst);
}
bool DataflowEdge::operator>=(DataflowEdge const &other) const {
  return std::tie(this->src, this->dst) >= std::tie(other.src, other.dst);
}
} // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::DataflowEdge>::operator()(
    ::FlexFlow::DataflowEdge const &x) const {
  size_t result = 0;
  result ^= std::hash<::FlexFlow::DataflowOutput>{}(x.src) + 0x9e3779b9 +
            (result << 6) + (result >> 2);
  result ^= std::hash<::FlexFlow::DataflowInput>{}(x.dst) + 0x9e3779b9 +
            (result << 6) + (result >> 2);
  return result;
}
} // namespace std

namespace FlexFlow {
std::string format_as(DataflowEdge const &x) {
  std::ostringstream oss;
  oss << "<DataflowEdge";
  oss << " src=" << x.src;
  oss << " dst=" << x.dst;
  oss << ">";
  return oss.str();
}
std::ostream &operator<<(std::ostream &s, DataflowEdge const &x) {
  return s << fmt::to_string(x);
}
} // namespace FlexFlow
