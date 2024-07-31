#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNLABELLED_INPUT_PATTERN_EDGE_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNLABELLED_INPUT_PATTERN_EDGE_H

#include "substitutions/unlabelled/input_pattern_edge.dtg.h"
#include "substitutions/unlabelled/pattern_input.dtg.h"
#include "substitutions/unlabelled/pattern_node.dtg.h"

namespace FlexFlow {

PatternInput get_src_input(InputPatternEdge const &);
PatternNode get_dst_node(InputPatternEdge const &);
int get_dst_idx(InputPatternEdge const &);

} // namespace FlexFlow

#endif
