// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/substitutions/include/substitutions/unlabelled/input_pattern_edge.struct.toml
/* proj-data
{
  "generated_from": "d0cc0e65c4e3feb2e9b8435947c99e5f"
}
*/

#include "substitutions/unlabelled/input_pattern_edge.dtg.h"

#include "utils/graph.h"

namespace FlexFlow {
InputPatternEdge::InputPatternEdge(::FlexFlow::InputMultiDiEdge const &raw_edge)
    : raw_edge(raw_edge) {}
bool InputPatternEdge::operator==(InputPatternEdge const &other) const {
  return std::tie(this->raw_edge) == std::tie(other.raw_edge);
}
bool InputPatternEdge::operator!=(InputPatternEdge const &other) const {
  return std::tie(this->raw_edge) != std::tie(other.raw_edge);
}
bool InputPatternEdge::operator<(InputPatternEdge const &other) const {
  return std::tie(this->raw_edge) < std::tie(other.raw_edge);
}
bool InputPatternEdge::operator>(InputPatternEdge const &other) const {
  return std::tie(this->raw_edge) > std::tie(other.raw_edge);
}
bool InputPatternEdge::operator<=(InputPatternEdge const &other) const {
  return std::tie(this->raw_edge) <= std::tie(other.raw_edge);
}
bool InputPatternEdge::operator>=(InputPatternEdge const &other) const {
  return std::tie(this->raw_edge) >= std::tie(other.raw_edge);
}
} // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::InputPatternEdge>::operator()(
    ::FlexFlow::InputPatternEdge const &x) const {
  size_t result = 0;
  result ^= std::hash<::FlexFlow::InputMultiDiEdge>{}(x.raw_edge) + 0x9e3779b9 +
            (result << 6) + (result >> 2);
  return result;
}
} // namespace std