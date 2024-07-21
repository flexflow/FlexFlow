#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_IMM_DOMINATORS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_IMM_DOMINATORS_H

#include "utils/graph/digraph/digraph_view.h"

namespace FlexFlow {

std::unordered_map<Node, std::optional<Node>>
    get_imm_dominators(DiGraphView const &);

} // namespace FlexFlow

#endif
