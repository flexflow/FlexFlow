// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/pcg/include/pcg/dataflow_graph/operator_added_result.struct.toml
/* proj-data
{
  "generated_from": "62224733c501773b41f1fc63a8677949"
}
*/

#include "pcg/dataflow_graph/operator_added_result.dtg.h"

#include "utils/fmt/vector.h"
#include "utils/graph.h"
#include <sstream>
#include <vector>

namespace FlexFlow {
OperatorAddedResult::OperatorAddedResult(
    ::FlexFlow::Node const &node,
    std::vector<::FlexFlow::MultiDiOutput> const &outputs)
    : node(node), outputs(outputs) {}
bool OperatorAddedResult::operator==(OperatorAddedResult const &other) const {
  return std::tie(this->node, this->outputs) ==
         std::tie(other.node, other.outputs);
}
bool OperatorAddedResult::operator!=(OperatorAddedResult const &other) const {
  return std::tie(this->node, this->outputs) !=
         std::tie(other.node, other.outputs);
}
bool OperatorAddedResult::operator<(OperatorAddedResult const &other) const {
  return std::tie(this->node, this->outputs) <
         std::tie(other.node, other.outputs);
}
bool OperatorAddedResult::operator>(OperatorAddedResult const &other) const {
  return std::tie(this->node, this->outputs) >
         std::tie(other.node, other.outputs);
}
bool OperatorAddedResult::operator<=(OperatorAddedResult const &other) const {
  return std::tie(this->node, this->outputs) <=
         std::tie(other.node, other.outputs);
}
bool OperatorAddedResult::operator>=(OperatorAddedResult const &other) const {
  return std::tie(this->node, this->outputs) >=
         std::tie(other.node, other.outputs);
}
} // namespace FlexFlow

namespace FlexFlow {
std::string format_as(OperatorAddedResult const &x) {
  std::ostringstream oss;
  oss << "<OperatorAddedResult";
  oss << " node=" << x.node;
  oss << " outputs=" << x.outputs;
  oss << ">";
  return oss.str();
}
std::ostream &operator<<(std::ostream &s, OperatorAddedResult const &x) {
  return s << fmt::to_string(x);
}
} // namespace FlexFlow
