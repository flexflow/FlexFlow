#include "utils/graph/upward_open_multidigraph/upward_open_multi_di_edge.h"
#include "utils/overload.h"

namespace FlexFlow {

OpenMultiDiEdge open_multidiedge_from_upward_open(UpwardOpenMultiDiEdge const &upward_e) {
  return upward_e.visit<OpenMultiDiEdge>(overload {
    [](MultiDiEdge const &e) { return OpenMultiDiEdge{e}; },
    [](OpenMultiDiEdge const &e) { return OpenMultiDiEdge{e}; },
  });
}

} // namespace FlexFlow
