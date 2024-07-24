#include "utils/graph/multidigraph/algorithms/get_directed_edge.h"

namespace FlexFlow {

DirectedEdge get_directed_edge(MultiDiGraphView const &g, MultiDiEdge const &e) {
  return DirectedEdge{g.get_multidiedge_src(e), g.get_multidiedge_dst(e)};
}

} // namespace FlexFlow
