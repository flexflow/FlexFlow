// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/substitutions/include/substitutions/unlabelled/match_additional_criterion.struct.toml
/* proj-data
{
  "generated_from": "2dff356c85dccda1fce8f714d41c6202"
}
*/

#include "substitutions/unlabelled/match_additional_criterion.dtg.h"

namespace FlexFlow {
MatchAdditionalCriterion::MatchAdditionalCriterion(
    std::function<bool(::FlexFlow::PatternNode const &,
                       ::FlexFlow::Node const &)> const &node_criterion,
    std::function<bool(::FlexFlow::PatternEdge const &,
                       ::FlexFlow::OpenMultiDiEdge const &)> const
        &edge_criterion)
    : node_criterion(node_criterion), edge_criterion(edge_criterion) {}
} // namespace FlexFlow
