#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNLABELLED_UNLABELLED_GRAPH_PATTERN_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNLABELLED_UNLABELLED_GRAPH_PATTERN_H

#include "substitutions/unlabelled/downward_open_pattern_edge.dtg.h"
#include "substitutions/unlabelled/pattern_edge.dtg.h"
#include "substitutions/unlabelled/pattern_node.dtg.h"
#include "substitutions/unlabelled/unlabelled_graph_pattern.dtg.h"
#include "substitutions/unlabelled/upward_open_pattern_edge.dtg.h"

namespace FlexFlow {

size_t num_nodes(UnlabelledGraphPattern const &);
bool is_singleton_pattern(UnlabelledGraphPattern const &);
std::unordered_set<PatternNode> get_nodes(UnlabelledGraphPattern const &);
std::unordered_set<PatternEdge> get_edges(UnlabelledGraphPattern const &);
std::vector<PatternNode>
    get_topological_ordering(UnlabelledGraphPattern const &);

std::unordered_set<UpwardOpenPatternEdge>
    get_incoming_edges(UnlabelledGraphPattern const &, PatternNode const &);
std::unordered_set<DownwardOpenPatternEdge>
    get_outgoing_edges(UnlabelledGraphPattern const &, PatternNode const &);

UnlabelledGraphPattern get_subgraph(UnlabelledGraphPattern const &,
                                    std::unordered_set<PatternNode> const &);

} // namespace FlexFlow

#endif
