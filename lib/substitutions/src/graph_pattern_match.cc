#include "substitutions/graph_pattern.h"
#include "utils/hash-utils.h"
#include <memory>

namespace FlexFlow {

GraphSplit split_pattern(OpenMultiDiGraphView const &pattern) {
  std::vector<Node> topological_ordering = get_topological_ordering(pattern);
  assert(topological_ordering.size() >= 2);

  int split_point = topological_ordering.size() / 2;
  auto split = vector_split(topological_ordering, split_point);
  std::unordered_set<Node> prefix(split.first.begin(), split.first.end());
  std::unordered_set<Node> postfix(split.second.begin(), split.second.end());
  return {prefix, postfix};
}

std::pair<OpenMultiDiGraphView, OpenMultiDiGraphView>
    apply_split(OpenMultiDiGraphView const &pattern, GraphSplit const &split) {
  return {get_subgraph(pattern, split.first),
          get_subgraph(pattern, split.second)};
}

/*
Given a match and a pattern split, gets the submatches in subpatterns.
*/
MatchSplit apply_split(OpenMultiDiGraphView const &pattern,
                       MultiDiGraphPatternMatch const &match,
                       GraphSplit const &split) {
  auto prefix = split.first;
  auto postfix = split.second;

  MatchSplit result;

  for (auto const &kv : match.node_assignment) {
    Node pattern_node = kv.first;
    Node graph_node = kv.second;
    if (contains(split.first, pattern_node)) {
      result.prefix_submatch.node_assignment.equate(pattern_node, graph_node);
    } else {
      assert(contains(split.second, pattern_node));
      result.postfix_submatch.node_assignment.equate(pattern_node, graph_node);
    }
  }

  auto edge_splits = get_edge_splits(pattern, split);

  std::function<void(OpenMultiDiEdge const &)> handle_edge =
      [&](OpenMultiDiEdge const &pattern_edge) -> void {
    OpenMultiDiEdge graph_edge = match.edge_assignment.at_l(pattern_edge);
    auto edge_nodes = get_nodes(pattern_edge);
    if (is_subseteq_of(edge_nodes, prefix)) {
      result.prefix_submatch.edge_assignment.equate(pattern_edge, graph_edge);
    } else if (is_subseteq_of(edge_nodes, postfix)) {
      result.postfix_submatch.edge_assignment.equate(pattern_edge, graph_edge);
    } else {
      assert(is_standard_edge(pattern_edge));
      auto standard_edge = mpark::get<MultiDiEdge>(pattern_edge);
      auto divided = edge_splits.at_l(standard_edge);
      handle_edge(divided.first);
      handle_edge(divided.second);
    }
  };

  for (auto const &kv : match.edge_assignment) {
    OpenMultiDiEdge pattern_edge = kv.first;
    handle_edge(pattern_edge);
  }

  return result;
}

bool is_singleton_pattern(OpenMultiDiGraphView const &pattern) {
  return num_nodes(pattern) == 1;
}

template <typename F>
bool pattern_matches(OpenMultiDiGraphView const &pattern,
                     OpenMultiDiGraphView const &graph,
                     MultiDiGraphPatternMatch const &match,
                     F const &additional_criterion) {
  if (is_singleton_pattern(pattern)) {
    Node pattern_node = get_only(get_nodes(pattern));
    Node graph_matched_node = match.node_assignment.at_l(pattern_node);
    if (!additional_criterion(pattern_node, graph_matched_node)) {
      return false;
    }
    for (OpenMultiDiEdge const &e : get_edges(pattern)) {
      OpenMultiDiEdge graph_matched_edge = match.edge_assignment.at_l(e);

      assert(is_input_edge(e) || is_output_edge(e));
      if (is_input_edge(e)) {
        InputMultiDiEdge input_edge = mpark::get<InputMultiDiEdge>(e);
        if (match.node_assignment.at_l(input_edge.dst) !=
                get_dst_node(graph_matched_edge) ||
            input_edge.dstIdx != get_dst_idx(graph_matched_edge)) {
          return false;
        }
      } else {
        OutputMultiDiEdge output_edge = mpark::get<OutputMultiDiEdge>(e);
        if (match.node_assignment.at_l(output_edge.src) !=
                get_src_node(graph_matched_edge) ||
            output_edge.srcIdx != get_src_idx(graph_matched_edge)) {
          return false;
        }
      }

      if (!additional_criterion(e, graph_matched_edge)) {
        return false;
      }
    }

    return true;
  }

  auto split = split_pattern(pattern);
  auto subpatterns = apply_split(pattern, split);
  auto submatches = apply_split(pattern, match, split);

  return pattern_matches(subpatterns.first,
                         graph,
                         submatches.prefix_submatch,
                         additional_criterion) &&
         pattern_matches(subpatterns.second,
                         graph,
                         submatches.postfix_submatch,
                         additional_criterion);
}

optional<MultiDiGraphPatternMatch>
    get_candidate_singleton_match(OpenMultiDiGraphView const &pattern,
                                  MultiDiGraphView const &graph,
                                  Node const &graph_node) {
  assert(is_singleton_pattern(pattern));

  Node pattern_node = get_only(get_nodes(pattern));

  MultiDiGraphPatternMatch match;
  match.node_assignment.equate(pattern_node, graph_node);

  auto incoming = get_incoming_edges_by_idx(graph, graph_node);
  auto outgoing = get_outgoing_edges_by_idx(graph, graph_node);
  for (OpenMultiDiEdge const &pattern_edge : get_edges(pattern)) {
    assert(is_input_edge(pattern_edge) || is_output_edge(pattern_edge));
    if (is_input_edge(pattern_edge)) {
      InputMultiDiEdge input_edge = mpark::get<InputMultiDiEdge>(pattern_edge);
      if (!contains_key(incoming, input_edge.dstIdx)) {
        return nullopt;
      }
      match.edge_assignment.equate(input_edge,
                                   get_only(incoming.at(input_edge.dstIdx)));
    } else {
      OutputMultiDiEdge output_edge =
          mpark::get<OutputMultiDiEdge>(pattern_edge);
      if (!contains_key(outgoing, output_edge.srcIdx)) {
        return nullopt;
      }
      match.edge_assignment.equate(output_edge,
                                   get_only(outgoing.at(output_edge.srcIdx)));
    }
  }

  return match;
}

optional<MultiDiGraphPatternMatch> unsplit_matches(
    MultiDiGraphPatternMatch const &prefix,
    MultiDiGraphPatternMatch const &postfix,
    bidict<MultiDiEdge, std::pair<OutputMultiDiEdge, InputMultiDiEdge>> const
        &edge_splits) {
  MultiDiGraphPatternMatch result;
  std::unordered_set<OpenMultiDiEdge> handled;
  for (auto const &kv : edge_splits) {
    MultiDiEdge standard_edge = kv.first;
    OutputMultiDiEdge output_edge = kv.second.first;
    InputMultiDiEdge input_edge = kv.second.second;
    handled.insert(output_edge);
    handled.insert(input_edge);

    OpenMultiDiEdge output_graph_edge =
        prefix.edge_assignment.at_l(output_edge);
    OpenMultiDiEdge input_graph_edge = postfix.edge_assignment.at_l(input_edge);
    if (output_graph_edge == input_graph_edge) {
      result.edge_assignment.equate(standard_edge, output_graph_edge);
    } else {
      return nullopt;
    }
  }

  for (auto const &kv :
       merge_maps(prefix.edge_assignment, postfix.edge_assignment)) {
    if (!contains(handled, kv.first)) {
      result.edge_assignment.equate(kv.first, kv.second);
    }
  }

  result.node_assignment =
      merge_maps(prefix.node_assignment, postfix.node_assignment);

  return result;
}

template <typename F>
std::unordered_set<MultiDiGraphPatternMatch>
    find_pattern_matches(OpenMultiDiGraphView const &pattern,
                         MultiDiGraphView const &graph,
                         F const &additional_criterion) {
  std::unordered_set<MultiDiGraphPatternMatch> matches;
  if (is_singleton_pattern(pattern)) {
    for (Node const &graph_node : get_nodes(graph)) {
      optional<MultiDiGraphPatternMatch> candidate =
          get_candidate_singleton_match(pattern, graph, graph_node);
      if (candidate.has_value() ||
          pattern_matches<F>(pattern, graph, candidate.value())) {
        matches.insert(candidate.value());
      }
    }
  } else {
    GraphSplit split = split_pattern(pattern);
    auto subpatterns = apply_split(pattern, split);
    auto prefix_matches =
        find_pattern_matches(subpatterns.first, graph, additional_criterion);
    auto postfix_matches =
        find_pattern_matches(subpatterns.first, graph, additional_criterion);
    auto edge_splits = get_edge_splits(pattern, split);
    for (MultiDiGraphPatternMatch const &prefix_match : prefix_matches) {
      for (MultiDiGraphPatternMatch const &postfix_match : postfix_matches) {
        optional<MultiDiGraphPatternMatch> unsplit =
            unsplit_matches(prefix_match, postfix_match, edge_splits);
        if (unsplit.has_value()) {
          matches.insert(unsplit.value());
        }
      }
    }
  }

  return matches;
}

} // namespace FlexFlow
