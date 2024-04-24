#include "substitutions/unlabelled/pattern_matching.h"
#include "substitutions/unlabelled/match_split.h"
#include "substitutions/unlabelled/unlabelled_graph_pattern.h"
#include "substitutions/unlabelled/pattern_edge.h"
#include <memory>
#include "substitutions/unlabelled/input_pattern_edge.h"
#include "substitutions/unlabelled/output_pattern_edge.h"
#include "substitutions/unlabelled/pattern_split.h"

namespace FlexFlow {

bool unlabelled_pattern_does_match(UnlabelledGraphPattern const &pattern,
                     OpenMultiDiGraphView const &graph,
                     MultiDiGraphPatternMatch const &match,
                     MatchAdditionalCriterion const &additional_criterion) {
  if (is_singleton_pattern(pattern)) {
    PatternNode pattern_node = get_only(get_nodes(pattern));
    Node matched_node = match.node_assignment.at_l(pattern_node);
    if (!additional_criterion.node_criterion(pattern_node,
                                             matched_node)) {
      return false;
    }
    for (PatternEdge const &e : get_edges(pattern)) {
      OpenMultiDiEdge matched_edge = match.edge_assignment.at_l(e);

      assert(is_input_edge(e) || is_output_edge(e));
      if (is_input_edge(e)) {
        if (is_output_edge(matched_edge)) {
          return false;
        }
        UpwardOpenMultiDiEdge matched_edge =
            narrow<UpwardOpenMultiDiEdge>(matched_edge).value();
        InputPatternEdge input_edge = require_input_edge(e);
        if (match.node_assignment.at_l(get_dst_node(input_edge)) !=
            get_dst_node(matched_edge)) {
          return false;
        }
      } else {
        if (is_input_edge(matched_edge)) {
          return false;
        }
        DownwardOpenMultiDiEdge matched_edge =
            narrow<DownwardOpenMultiDiEdge>(matched_edge).value();
        OutputPatternEdge output_edge = require_output_edge(e);
        if (match.node_assignment.at_l(get_src_node(output_edge)) !=
            get_src_node(matched_edge)) {
          return false;
        }
      }

      if (!additional_criterion.edge_criterion(e, matched_edge)) {
        return false;
      }
    }

    return true;
  }

  PatternSplit split = find_even_split(pattern);
  std::pair<UnlabelledGraphPattern, UnlabelledGraphPattern> subpatterns = apply_split(pattern, split);
  auto submatches = apply_split(pattern, match, split);

  return unlabelled_pattern_does_match(subpatterns.first,
                         graph,
                         submatches.prefix_submatch,
                         additional_criterion) &&
         unlabelled_pattern_does_match(subpatterns.second,
                         graph,
                         submatches.postfix_submatch,
                         additional_criterion);
}


} // namespace FlexFlow
