#include "substitutions/unlabelled/input_pattern_edge.h"

namespace FlexFlow {

PatternNode get_dst_node(InputPatternEdge const &e) {
  return PatternNode{e.raw_edge.dst};
}

} // namespace FlexFlow
