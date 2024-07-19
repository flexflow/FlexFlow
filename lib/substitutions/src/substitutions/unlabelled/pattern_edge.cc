#include "substitutions/unlabelled/pattern_edge.h"
#include "substitutions/unlabelled/input_pattern_edge.h"
#include "substitutions/unlabelled/standard_pattern_edge.h"
#include "utils/containers.h"
#include "utils/overload.h"

namespace FlexFlow {

std::unordered_set<PatternNode> get_nodes(PatternEdge const &e) {
  return e.visit<std::unordered_set<PatternNode>>(overload{
      [](InputPatternEdge const &ee) {
        return std::unordered_set<PatternNode>{get_dst_node(ee)};
      },
      [](StandardPatternEdge const &ee) {
        return std::unordered_set<PatternNode>{
            get_src_node(ee),
            get_dst_node(ee),
        };
      },
  });
}

bool is_standard_edge(PatternEdge const &e) {
  return e.has<StandardPatternEdge>();
}

bool is_input_edge(PatternEdge const &e) {
  return e.has<InputPatternEdge>();
}

StandardPatternEdge require_standard_edge(PatternEdge const &e) {
  assert(is_standard_edge(e));
  return e.get<StandardPatternEdge>();
}

InputPatternEdge require_input_edge(PatternEdge const &e) {
  assert(is_input_edge(e));
  return e.get<InputPatternEdge>();
}

PatternEdge pattern_edge_from_input_edge(InputPatternEdge const &e) {
  return PatternEdge{e};
}

PatternEdge pattern_edge_from_standard_edge(StandardPatternEdge const &e) {
  return PatternEdge{e};
}

PatternEdge
    pattern_edge_from_raw_open_dataflow_edge(OpenDataflowEdge const &e) {
  return e.visit<PatternEdge>(overload{
      [](DataflowInputEdge const &ee) {
        return PatternEdge{InputPatternEdge{ee}};
      },
      [](DataflowEdge const &ee) {
        return PatternEdge{StandardPatternEdge{ee}};
      },
  });
}

} // namespace FlexFlow
