#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_PATTERN_EDGE_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_PATTERN_EDGE_H

#include "substitutions/unlabelled/input_pattern_edge.dtg.h"
#include "substitutions/unlabelled/pattern_edge.dtg.h"
#include "substitutions/unlabelled/pattern_node.dtg.h"
#include "substitutions/unlabelled/standard_pattern_edge.dtg.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_edge.dtg.h"
#include <unordered_set>

namespace FlexFlow {

PatternNode get_dst_node(PatternEdge const &);

std::unordered_set<PatternNode> get_nodes(PatternEdge const &);
bool is_input_edge(PatternEdge const &);
bool is_standard_edge(PatternEdge const &);

StandardPatternEdge require_standard_edge(PatternEdge const &);
InputPatternEdge require_input_edge(PatternEdge const &);

PatternEdge pattern_edge_from_input_edge(InputPatternEdge const &);
PatternEdge pattern_edge_from_standard_edge(StandardPatternEdge const &);

PatternEdge pattern_edge_from_raw_open_dataflow_edge(OpenDataflowEdge const &);

} // namespace FlexFlow

#endif
