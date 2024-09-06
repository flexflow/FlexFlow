#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_COMPLETE_BIPARTITE_COMPOSITE_IS_COMPLETE_BIPARTITE_DIGRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_COMPLETE_BIPARTITE_COMPOSITE_IS_COMPLETE_BIPARTITE_DIGRAPH_H

#include "utils/graph/digraph/digraph_view.h"

namespace FlexFlow {

bool is_complete_bipartite_digraph(DiGraphView const &);
bool is_complete_bipartite_digraph(DiGraphView const &,
                                   std::unordered_set<Node> const &srcs);

} // namespace FlexFlow

#endif
