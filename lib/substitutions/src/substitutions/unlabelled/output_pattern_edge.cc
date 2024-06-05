#include "substitutions/unlabelled/output_pattern_edge.h"

namespace FlexFlow {

PatternNode get_src_node(OutputPatternEdge const &e) {
  return PatternNode{e.raw_edge.src};
}

} // namespace FlexFlow
