// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/substitutions/include/substitutions/unlabelled/pattern_node.struct.toml
/* proj-data
{
  "generated_from": "a0e58ade010a9b250d2c1c378fde2639"
}
*/

#include "substitutions/unlabelled/pattern_node.dtg.h"

#include "utils/graph.h"

namespace FlexFlow {
PatternNode::PatternNode(::FlexFlow::Node const &raw_node)
    : raw_node(raw_node) {}
bool PatternNode::operator==(PatternNode const &other) const {
  return std::tie(this->raw_node) == std::tie(other.raw_node);
}
bool PatternNode::operator!=(PatternNode const &other) const {
  return std::tie(this->raw_node) != std::tie(other.raw_node);
}
bool PatternNode::operator<(PatternNode const &other) const {
  return std::tie(this->raw_node) < std::tie(other.raw_node);
}
bool PatternNode::operator>(PatternNode const &other) const {
  return std::tie(this->raw_node) > std::tie(other.raw_node);
}
bool PatternNode::operator<=(PatternNode const &other) const {
  return std::tie(this->raw_node) <= std::tie(other.raw_node);
}
bool PatternNode::operator>=(PatternNode const &other) const {
  return std::tie(this->raw_node) >= std::tie(other.raw_node);
}
} // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::PatternNode>::operator()(
    FlexFlow::PatternNode const &x) const {
  size_t result = 0;
  result ^= std::hash<::FlexFlow::Node>{}(x.raw_node) + 0x9e3779b9 +
            (result << 6) + (result >> 2);
  return result;
}
} // namespace std
