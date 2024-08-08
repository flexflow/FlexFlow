#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNLABELLED_PATTERN_NODE_OUTPUT_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNLABELLED_PATTERN_NODE_OUTPUT_H

#include "substitutions/unlabelled/pattern_node.dtg.h"
#include "substitutions/unlabelled/pattern_node_output.dtg.h"
namespace FlexFlow {

PatternNode get_src_node(PatternNodeOutput const &);
int get_idx(PatternNodeOutput const &);

} // namespace FlexFlow

#endif
