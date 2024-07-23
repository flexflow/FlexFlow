#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_COMPLETE_BIPARTITE_COMPOSITE_COMPLETE_BIPARTITE_COMPOSITE_DECOMPOSITION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_COMPLETE_BIPARTITE_COMPOSITE_COMPLETE_BIPARTITE_COMPOSITE_DECOMPOSITION_H

#include "utils/graph/digraph/algorithms/complete_bipartite_composite/complete_bipartite_composite_decomposition.dtg.h"
#include <optional>

namespace FlexFlow {

std::optional<BipartiteComponent> get_component_containing_node_in_head(CompleteBipartiteCompositeDecomposition const &, Node const &);
std::optional<BipartiteComponent> get_component_containing_node_in_tail(CompleteBipartiteCompositeDecomposition const &, Node const &);
std::unordered_set<std::unordered_set<Node>> get_head_subcomponents(CompleteBipartiteCompositeDecomposition const &);
std::unordered_set<std::unordered_set<Node>> get_tail_subcomponents(CompleteBipartiteCompositeDecomposition const &);

} // namespace FlexFlow

#endif
