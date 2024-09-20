#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_IS_ACYCLIC_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_IS_ACYCLIC_H

#include "utils/graph/digraph/digraph_view.h"

namespace FlexFlow {

std::optional<bool> is_acyclic(DiGraphView const &);

} // namespace FlexFlow

#endif
