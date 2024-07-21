#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_IMM_POST_DOMINATOR_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_IMM_POST_DOMINATOR_H

#include "utils/graph/digraph/digraph_view.h"

namespace FlexFlow {

std::optional<Node> get_imm_post_dominator(DiGraphView const &, Node const &);
std::optional<Node> get_imm_post_dominator(DiGraphView const &,
                                           std::unordered_set<Node> const &);

} // namespace FlexFlow

#endif
