#include "substitutions/unlabelled/input_pattern_edge.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_edge.h"

namespace FlexFlow {

PatternInput get_src_input(InputPatternEdge const &) {
  NOT_IMPLEMENTED();
}

PatternNode get_dst_node(InputPatternEdge const &e) {
  return PatternNode{get_open_dataflow_edge_dst_node(e.raw_edge)};
}

} // namespace FlexFlow
