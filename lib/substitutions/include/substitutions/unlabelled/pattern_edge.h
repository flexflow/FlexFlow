#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_PATTERN_EDGE_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_PATTERN_EDGE_H

#include "substitutions/unlabelled/closed_pattern_edge.dtg.h"
#include "substitutions/unlabelled/input_pattern_edge.dtg.h"
#include "substitutions/unlabelled/output_pattern_edge.dtg.h"
#include "substitutions/unlabelled/pattern_edge.dtg.h"
#include "substitutions/unlabelled/pattern_node.dtg.h"

namespace FlexFlow {

std::unordered_set<PatternNode> get_nodes(PatternEdge const &);
bool is_closed_edge(PatternEdge const &);
bool is_input_edge(PatternEdge const &);
bool is_output_edge(PatternEdge const &);

ClosedPatternEdge require_closed_edge(PatternEdge const &);
InputPatternEdge require_input_edge(PatternEdge const &);
OutputPatternEdge require_output_edge(PatternEdge const &);

PatternEdge pattern_edge_from_input_edge(InputPatternEdge const &);
PatternEdge pattern_edge_from_output_edge(OutputPatternEdge const &);
PatternEdge pattern_edge_from_closed_edge(ClosedPatternEdge const &);

} // namespace FlexFlow

#endif
