#include "substitutions/unlabelled/pattern_node_output.h"

namespace FlexFlow {

PatternNode get_src_node(PatternNodeOutput const &o) {
  return PatternNode{o.raw_dataflow_output.node};
}

int get_idx(PatternNodeOutput const &o) {
  return o.raw_dataflow_output.idx;
}

} // namespace FlexFlow
