#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_COMPLETE_BIPARTITE_COMPOSITE_GET_CBC_DECOMPOSITION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_COMPLETE_BIPARTITE_COMPOSITE_GET_CBC_DECOMPOSITION_H

#include "utils/graph/digraph/algorithms/complete_bipartite_composite/complete_bipartite_composite_decomposition.dtg.h"
#include "utils/graph/digraph/digraph_view.h"

namespace FlexFlow {

std::optional<CompleteBipartiteCompositeDecomposition>
    get_cbc_decomposition(DiGraphView const &);

} // namespace FlexFlow

#endif
