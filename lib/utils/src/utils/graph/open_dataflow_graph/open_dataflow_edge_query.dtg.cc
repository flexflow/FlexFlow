// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/utils/include/utils/graph/open_dataflow_graph/open_dataflow_edge_query.struct.toml
/* proj-data
{
  "generated_from": "661c106abdb03bf6cc434d87cfafefb5"
}
*/

#include "utils/graph/open_dataflow_graph/open_dataflow_edge_query.dtg.h"

#include <sstream>

namespace FlexFlow {
OpenDataflowEdgeQuery::OpenDataflowEdgeQuery(
    ::FlexFlow::DataflowInputEdgeQuery const &input_edge_query,
    ::FlexFlow::DataflowEdgeQuery const &standard_edge_query)
    : input_edge_query(input_edge_query),
      standard_edge_query(standard_edge_query) {}
bool OpenDataflowEdgeQuery::operator==(
    OpenDataflowEdgeQuery const &other) const {
  return std::tie(this->input_edge_query, this->standard_edge_query) ==
         std::tie(other.input_edge_query, other.standard_edge_query);
}
bool OpenDataflowEdgeQuery::operator!=(
    OpenDataflowEdgeQuery const &other) const {
  return std::tie(this->input_edge_query, this->standard_edge_query) !=
         std::tie(other.input_edge_query, other.standard_edge_query);
}
bool OpenDataflowEdgeQuery::operator<(
    OpenDataflowEdgeQuery const &other) const {
  return std::tie(this->input_edge_query, this->standard_edge_query) <
         std::tie(other.input_edge_query, other.standard_edge_query);
}
bool OpenDataflowEdgeQuery::operator>(
    OpenDataflowEdgeQuery const &other) const {
  return std::tie(this->input_edge_query, this->standard_edge_query) >
         std::tie(other.input_edge_query, other.standard_edge_query);
}
bool OpenDataflowEdgeQuery::operator<=(
    OpenDataflowEdgeQuery const &other) const {
  return std::tie(this->input_edge_query, this->standard_edge_query) <=
         std::tie(other.input_edge_query, other.standard_edge_query);
}
bool OpenDataflowEdgeQuery::operator>=(
    OpenDataflowEdgeQuery const &other) const {
  return std::tie(this->input_edge_query, this->standard_edge_query) >=
         std::tie(other.input_edge_query, other.standard_edge_query);
}
} // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::OpenDataflowEdgeQuery>::operator()(
    ::FlexFlow::OpenDataflowEdgeQuery const &x) const {
  size_t result = 0;
  result ^=
      std::hash<::FlexFlow::DataflowInputEdgeQuery>{}(x.input_edge_query) +
      0x9e3779b9 + (result << 6) + (result >> 2);
  result ^= std::hash<::FlexFlow::DataflowEdgeQuery>{}(x.standard_edge_query) +
            0x9e3779b9 + (result << 6) + (result >> 2);
  return result;
}
} // namespace std

namespace FlexFlow {
std::string format_as(OpenDataflowEdgeQuery const &x) {
  std::ostringstream oss;
  oss << "<OpenDataflowEdgeQuery";
  oss << " input_edge_query=" << x.input_edge_query;
  oss << " standard_edge_query=" << x.standard_edge_query;
  oss << ">";
  return oss.str();
}
std::ostream &operator<<(std::ostream &s, OpenDataflowEdgeQuery const &x) {
  return s << fmt::to_string(x);
}
} // namespace FlexFlow
