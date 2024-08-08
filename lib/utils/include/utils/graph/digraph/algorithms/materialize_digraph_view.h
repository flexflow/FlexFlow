#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_MATERIALIZE_DIGRAPH_VIEW_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_MATERIALIZE_DIGRAPH_VIEW_H

#include "utils/graph/digraph/digraph.h"

namespace FlexFlow {

void materialize_digraph_view(DiGraph &, DiGraphView const &);

template <typename Impl>
DiGraph materialize_digraph_view(DiGraphView const &g) {
  DiGraph result = DiGraph::create<Impl>();
  materialize_digraph_view(result, g);
  return result;
}

} // namespace FlexFlow

#endif
