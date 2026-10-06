#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNLABELLED_PATTERN_MATCHING_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNLABELLED_PATTERN_MATCHING_H

#include "substitutions/unlabelled/match_additional_criterion.dtg.h"
#include "substitutions/unlabelled/standard_pattern_edge.dtg.h"
#include "substitutions/unlabelled/unlabelled_graph_pattern.dtg.h"
#include "substitutions/unlabelled/unlabelled_kwarg_dataflow_graph_pattern_match.dtg.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/open_kwarg_dataflow_subgraph_result.dtg.h"

namespace FlexFlow {

OpenKwargDataflowSubgraphResult<int, TensorSlotName> subgraph_matched(
    OpenKwargDataflowGraphView<int, TensorSlotName> const &graph,
    UnlabelledKwargDataflowGraphPatternMatch const &match);

bool pattern_matches_subgraph_under(
    UnlabelledGraphPattern const &pattern,
    OpenKwargDataflowGraphView<int, TensorSlotName> const &subgraph,
    bidict<OpenKwargDataflowValue<int, TensorSlotName>,
           KwargDataflowGraphInput<int>> const
        &full_graph_values_to_subgraph_inputs,
    UnlabelledKwargDataflowGraphPatternMatch const &match,
    MatchAdditionalCriterion const &additional_criterion);

bool unlabelled_pattern_does_match(
    UnlabelledGraphPattern const &pattern,
    OpenKwargDataflowGraphView<int, TensorSlotName> const &graph,
    UnlabelledKwargDataflowGraphPatternMatch const &match,
    MatchAdditionalCriterion const &additional_criterion);

bool unlabelled_pattern_does_match(
    UnlabelledGraphPattern const &pattern,
    OpenKwargDataflowGraphView<int, TensorSlotName> const &graph,
    UnlabelledKwargDataflowGraphPatternMatch const &match,
    MatchAdditionalCriterion const &additional_criterion,
    std::set<StandardPatternEdge> const &hint_edges);

} // namespace FlexFlow

#endif
