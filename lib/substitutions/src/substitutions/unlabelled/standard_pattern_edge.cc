#include "substitutions/unlabelled/standard_pattern_edge.h"

namespace FlexFlow {

PatternNode get_src_node(StandardPatternEdge const &e) {
  return PatternNode{e.raw_edge.src.node};
}

PatternNode get_dst_node(StandardPatternEdge const &e) {
  return PatternNode{e.raw_edge.dst.node};
}

int get_src_idx(StandardPatternEdge const &e) {
  return e.raw_edge.src.idx;
}

int get_dst_idx(StandardPatternEdge const &e) {
  return e.raw_edge.dst.idx;
}

} // namespace FlexFlow
