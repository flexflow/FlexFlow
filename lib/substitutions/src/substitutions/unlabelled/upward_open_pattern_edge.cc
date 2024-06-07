#include "substitutions/unlabelled/upward_open_pattern_edge.h"

namespace FlexFlow {

int get_dst_idx(UpwardOpenPatternEdge const &e) {
  return get_src_idx(e.raw_edge);
}

} // namespace FlexFlow
