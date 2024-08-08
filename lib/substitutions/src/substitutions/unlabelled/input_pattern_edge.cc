#include "substitutions/unlabelled/input_pattern_edge.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_edge.h"

namespace FlexFlow {

PatternInput get_src_input(InputPatternEdge const &e) {
  return PatternInput{e.raw_edge.src};
}

PatternNode get_dst_node(InputPatternEdge const &e) {
  return PatternNode{e.raw_edge.dst.node};
}

int get_dst_idx(InputPatternEdge const &e) {
  return e.raw_edge.dst.idx;
}

} // namespace FlexFlow
