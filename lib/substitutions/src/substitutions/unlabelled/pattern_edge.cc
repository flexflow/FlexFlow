#include "substitutions/unlabelled/pattern_edge.h"
#include "utils/containers.h"

namespace FlexFlow {

std::unordered_set<PatternNode> get_nodes(PatternEdge const &e) {
  return transform(get_nodes(e.raw_edge),
                   [](Node const &n) { return PatternNode{n}; });
}

bool is_standard_edge(PatternEdge const &e) {
  return is_standard_edge(e.raw_edge);
}

bool is_input_edge(PatternEdge const &e) {
  return is_input_edge(e.raw_edge);
}

bool is_output_edge(PatternEdge const &e) {
  return is_output_edge(e.raw_edge);
}

ClosedPatternEdge require_closed_edge(PatternEdge const &e) {
  assert(is_closed_edge(e));
  return ClosedPatternEdge{std::get<MultiDiEdge>(e.raw_edge)};
}

InputPatternEdge require_input_edge(PatternEdge const &e) {
  assert(is_input_edge(e));
  return InputPatternEdge{std::get<InputMultiDiEdge>(e.raw_edge)};
}

OutputPatternEdge require_output_edge(PatternEdge const &e) {
  assert(is_output_edge(e));
  return OutputPatternEdge{std::get<OutputMultiDiEdge>(e.raw_edge)};
}

PatternEdge pattern_edge_from_input_edge(InputPatternEdge const &e) {
  return PatternEdge{OpenMultiDiEdge{e.raw_edge}};
}

PatternEdge pattern_edge_from_output_edge(OutputPatternEdge const &e) {
  return PatternEdge{OpenMultiDiEdge{e.raw_edge}};
}

PatternEdge pattern_edge_from_closed_edge(ClosedPatternEdge const &e) {
  return PatternEdge{OpenMultiDiEdge{e.raw_edge}};
}

} // namespace FlexFlow
