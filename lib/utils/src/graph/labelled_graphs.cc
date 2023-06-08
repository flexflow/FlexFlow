#include "utils/graph/labelled_graphs.h"

namespace FlexFlow {

MultiDiOutput get_output(MultiDiEdge const &e) { return {e.src, e.srcIdx}; }

MultiDiInput get_input(MultiDiEdge const &e) { return {e.dst, e.dstIdx}; }

} // namespace FlexFlow
