// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/utils/include/utils/graph/dataflow_graph/dataflow_edge_query.struct.toml
/* proj-data
{
  "generated_from": "111e640382a80b659bc33dd86a416ded"
}
*/

#include "utils/graph/dataflow_graph/dataflow_edge_query.dtg.h"

#include <sstream>

namespace FlexFlow {
DataflowEdgeQuery::DataflowEdgeQuery(
    ::FlexFlow::query_set<::FlexFlow::Node> const &src_nodes,
    ::FlexFlow::query_set<int> const &src_idxs,
    ::FlexFlow::query_set<::FlexFlow::Node> const &dst_nodes,
    ::FlexFlow::query_set<int> const &dst_idxs)
    : src_nodes(src_nodes), src_idxs(src_idxs), dst_nodes(dst_nodes),
      dst_idxs(dst_idxs) {}
bool DataflowEdgeQuery::operator==(DataflowEdgeQuery const &other) const {
  return std::tie(this->src_nodes,
                  this->src_idxs,
                  this->dst_nodes,
                  this->dst_idxs) ==
         std::tie(
             other.src_nodes, other.src_idxs, other.dst_nodes, other.dst_idxs);
}
bool DataflowEdgeQuery::operator!=(DataflowEdgeQuery const &other) const {
  return std::tie(this->src_nodes,
                  this->src_idxs,
                  this->dst_nodes,
                  this->dst_idxs) !=
         std::tie(
             other.src_nodes, other.src_idxs, other.dst_nodes, other.dst_idxs);
}
bool DataflowEdgeQuery::operator<(DataflowEdgeQuery const &other) const {
  return std::tie(
             this->src_nodes, this->src_idxs, this->dst_nodes, this->dst_idxs) <
         std::tie(
             other.src_nodes, other.src_idxs, other.dst_nodes, other.dst_idxs);
}
bool DataflowEdgeQuery::operator>(DataflowEdgeQuery const &other) const {
  return std::tie(
             this->src_nodes, this->src_idxs, this->dst_nodes, this->dst_idxs) >
         std::tie(
             other.src_nodes, other.src_idxs, other.dst_nodes, other.dst_idxs);
}
bool DataflowEdgeQuery::operator<=(DataflowEdgeQuery const &other) const {
  return std::tie(this->src_nodes,
                  this->src_idxs,
                  this->dst_nodes,
                  this->dst_idxs) <=
         std::tie(
             other.src_nodes, other.src_idxs, other.dst_nodes, other.dst_idxs);
}
bool DataflowEdgeQuery::operator>=(DataflowEdgeQuery const &other) const {
  return std::tie(this->src_nodes,
                  this->src_idxs,
                  this->dst_nodes,
                  this->dst_idxs) >=
         std::tie(
             other.src_nodes, other.src_idxs, other.dst_nodes, other.dst_idxs);
}
} // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::DataflowEdgeQuery>::operator()(
    ::FlexFlow::DataflowEdgeQuery const &x) const {
  size_t result = 0;
  result ^= std::hash<::FlexFlow::query_set<::FlexFlow::Node>>{}(x.src_nodes) +
            0x9e3779b9 + (result << 6) + (result >> 2);
  result ^= std::hash<::FlexFlow::query_set<int>>{}(x.src_idxs) + 0x9e3779b9 +
            (result << 6) + (result >> 2);
  result ^= std::hash<::FlexFlow::query_set<::FlexFlow::Node>>{}(x.dst_nodes) +
            0x9e3779b9 + (result << 6) + (result >> 2);
  result ^= std::hash<::FlexFlow::query_set<int>>{}(x.dst_idxs) + 0x9e3779b9 +
            (result << 6) + (result >> 2);
  return result;
}
} // namespace std

namespace FlexFlow {
std::string format_as(DataflowEdgeQuery const &x) {
  std::ostringstream oss;
  oss << "<DataflowEdgeQuery";
  oss << " src_nodes=" << x.src_nodes;
  oss << " src_idxs=" << x.src_idxs;
  oss << " dst_nodes=" << x.dst_nodes;
  oss << " dst_idxs=" << x.dst_idxs;
  oss << ">";
  return oss.str();
}
std::ostream &operator<<(std::ostream &s, DataflowEdgeQuery const &x) {
  return s << fmt::to_string(x);
}
} // namespace FlexFlow
