#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNLABELLED_UNLABELLED_GRAPH_PATTERN_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNLABELLED_UNLABELLED_GRAPH_PATTERN_H

#include "substitutions/unlabelled/pattern_edge.dtg.h"
#include "substitutions/unlabelled/pattern_value.dtg.h"
#include "substitutions/unlabelled/pattern_node.dtg.h"
#include "substitutions/unlabelled/unlabelled_graph_pattern.dtg.h"
#include "substitutions/unlabelled/pattern_input.dtg.h"
#include "substitutions/unlabelled/unlabelled_graph_pattern_subgraph_result.dtg.h"

namespace FlexFlow {

size_t num_nodes(UnlabelledGraphPattern const &);
bool is_singleton_pattern(UnlabelledGraphPattern const &);
std::unordered_set<PatternNode> get_nodes(UnlabelledGraphPattern const &);
std::unordered_set<PatternValue> get_values(UnlabelledGraphPattern const &);
// std::unordered_set<PatternValueUse> get_value_uses(UnlabelledGraphPattern const &, PatternValue const &);
std::vector<PatternNode>
    get_topological_ordering(UnlabelledGraphPattern const &);

std::unordered_set<PatternInput> get_inputs(UnlabelledGraphPattern const &);

std::unordered_set<PatternEdge> get_edges(UnlabelledGraphPattern const &);

std::vector<PatternValue>
    get_inputs_to_pattern_node(UnlabelledGraphPattern const &, PatternNode const &);
std::vector<PatternValue>
    get_outputs_from_pattern_node(UnlabelledGraphPattern const &, PatternNode const &);

UnlabelledGraphPatternSubgraphResult get_subgraph(UnlabelledGraphPattern const &,
                                                  std::unordered_set<PatternNode> const &);

} // namespace FlexFlow

#endif
