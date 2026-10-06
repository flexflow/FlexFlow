#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNLABELLED_UNLABELLED_GRAPH_PATTERN_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNLABELLED_UNLABELLED_GRAPH_PATTERN_H

#include "substitutions/unlabelled/pattern_edge.dtg.h"
#include "substitutions/unlabelled/pattern_input.dtg.h"
#include "substitutions/unlabelled/pattern_node.dtg.h"
#include "substitutions/unlabelled/pattern_split.dtg.h"
#include "substitutions/unlabelled/pattern_value.dtg.h"
#include "substitutions/unlabelled/unlabelled_graph_pattern.dtg.h"
#include "substitutions/unlabelled/unlabelled_graph_pattern_subgraph_result.dtg.h"

namespace FlexFlow {

size_t num_nodes(UnlabelledGraphPattern const &);
bool is_singleton_pattern(UnlabelledGraphPattern const &);
std::set<PatternNode> get_pattern_nodes(UnlabelledGraphPattern const &);
std::set<PatternValue> get_pattern_values(UnlabelledGraphPattern const &);
std::vector<PatternNode>
    get_topological_ordering(UnlabelledGraphPattern const &);

std::set<PatternInput> get_pattern_inputs(UnlabelledGraphPattern const &);

std::set<StandardPatternEdge>
    get_pattern_edges_across_topological_split(UnlabelledGraphPattern const &,
                                               PatternSplit const &);

std::set<PatternEdge> get_pattern_edges(UnlabelledGraphPattern const &);

std::map<TensorSlotName, PatternValue>
    get_inputs_to_pattern_node(UnlabelledGraphPattern const &,
                               PatternNode const &);
std::map<TensorSlotName, PatternValue>
    get_outputs_from_pattern_node(UnlabelledGraphPattern const &,
                                  PatternNode const &);

UnlabelledGraphPatternSubgraphResult
    get_pattern_subgraph(UnlabelledGraphPattern const &,
                         std::set<PatternNode> const &);

} // namespace FlexFlow

#endif
