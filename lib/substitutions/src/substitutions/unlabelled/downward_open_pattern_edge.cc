#include "substitutions/unlabelled/downward_open_pattern_edge.h"

namespace FlexFlow {

int get_src_idx(DownwardOpenPatternEdge const &e) {
  return get_src_idx(e.raw_edge);
}

} // namespace FlexFlow
