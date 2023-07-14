#include "substitutions/graph_pattern.h"
#include "utils/hash-utils.h"
#include <memory>

namespace FlexFlow {
namespace substitutions {

DiGraphPatternMatch narrow_match(DiGraphPatternMatch const &match,
                                 IOpenMultiDiGraphView const &pattern) {
  DiGraphPatternMatch result;
  std::unordered_set<Node> nodes = get_nodes(pattern);
  for (auto const &kv : match.nodeAssignment) {
    Node pattern_node = kv.first;
    if (contains(nodes, pattern_node)) {
      result.nodeAssignment.equate(kv.first, kv.second);
    }
  }

  std::unordered_set<OpenMultiDiEdge> edges = get_edges(pattern);
  for (auto const &kv : match.edgeAssignment) {
    OpenMultiDiEdge pattern_edge = kv.first;
    if (contains(edges, pattern_edge)) {
      result.edgeAssignment.equate(kv.first, kv.second);
    }
  }

  return result;
}

GraphSplit split_pattern(IOpenMultiDiGraphView const &pattern) {
  std::vector<Node> topological_ordering = get_topological_ordering(pattern);
  assert(topological_ordering.size() >= 2);

  int split_point = topological_ordering.size() / 2;
  auto split = vector_split(topological_ordering, split_point);
  std::unordered_set<Node> prefix(split.first.begin(), split.first.end());
  std::unordered_set<Node> postfix(split.second.begin(), split.second.end());
  return {prefix, postfix};
}

std::pair<std::unique_ptr<IOpenMultiDiGraphView>,
          std::unique_ptr<IOpenMultiDiGraphView>>
    apply_split(IOpenMultiDiGraphView const &pattern, GraphSplit const &split) {
  return {unsafe_view_as_subgraph(pattern, split.first),
          unsafe_view_as_subgraph(pattern, split.second)};
}

std::unordered_set<Node> get_nodes(OpenMultiDiEdge const &pattern_edge) {
  if (is_input_edge(pattern_edge)) {
    return {mpark::get<InputMultiDiEdge>(pattern_edge).dst};
  } else if (is_output_edge(pattern_edge)) {
    return {mpark::get<OutputMultiDiEdge>(pattern_edge).src};
  } else {
    assert(is_standard_edge(pattern_edge));
    auto standard_edge = mpark::get<MultiDiEdge>(pattern_edge);
    return {standard_edge.src, standard_edge.dst};
  }
}

bidict<MultiDiEdge, std::pair<OutputMultiDiEdge, InputMultiDiEdge>>
    get_edge_splits(IOpenMultiDiGraphView const &pattern,
                    GraphSplit const &split) {
  auto prefix = split.first;
  auto postfix = split.second;

  bidict<MultiDiEdge, std::pair<OutputMultiDiEdge, InputMultiDiEdge>> result;

  for (OpenMultiDiEdge const &pattern_edge : get_edges(pattern)) {
    if (!is_standard_edge(pattern_edge)) {
      continue;
    }

    auto standard_edge = mpark::get<MultiDiEdge>(pattern_edge);
    if (is_subseteq_of(get_nodes(standard_edge), prefix) ||
        is_subseteq_of(get_nodes(standard_edge), postfix)) {
      continue;
    }

    auto divided = split_edge(standard_edge);
    result.equate(standard_edge, divided);
  }

  return result;
}

MatchSplit apply_split(IOpenMultiDiGraphView const &pattern,
                       DiGraphPatternMatch const &match,
                       GraphSplit const &split) {
  auto prefix = split.first;
  auto postfix = split.second;

  MatchSplit result;

  for (auto const &kv : match.nodeAssignment) {
    Node pattern_node = kv.first;
    Node graph_node = kv.second;
    if (contains(split.first, pattern_node)) {
      result.prefix_submatch.nodeAssignment.equate(pattern_node, graph_node);
    } else {
      assert(contains(split.second, pattern_node));
      result.postfix_submatch.nodeAssignment.equate(pattern_node, graph_node);
    }
  }

  auto edge_splits = get_edge_splits(pattern, split);

  std::function<void(OpenMultiDiEdge const &)> handle_edge =
      [&](OpenMultiDiEdge const &pattern_edge) -> void {
    MultiDiEdge graph_edge = match.edgeAssignment.at_l(pattern_edge);
    auto edge_nodes = get_nodes(pattern_edge);
    if (is_subseteq_of(edge_nodes, prefix)) {
      result.prefix_submatch.edgeAssignment.equate(pattern_edge, graph_edge);
    } else if (is_subseteq_of(edge_nodes, postfix)) {
      result.postfix_submatch.edgeAssignment.equate(pattern_edge, graph_edge);
    } else {
      assert(is_standard_edge(pattern_edge));
      auto standard_edge = mpark::get<MultiDiEdge>(pattern_edge);
      auto divided = edge_splits.at_l(standard_edge);
      handle_edge(divided.first);
      handle_edge(divided.second);
    }
  };

  for (auto const &kv : match.edgeAssignment) {
    OpenMultiDiEdge pattern_edge = kv.first;
    handle_edge(pattern_edge);
  }

  return result;
}

bool is_singleton_pattern(IOpenMultiDiGraphView const &pattern) {
  return num_nodes(pattern) == 1;
}

template <typename F>
bool pattern_matches(IOpenMultiDiGraphView const &pattern,
                     IMultiDiGraph const &graph,
                     DiGraphPatternMatch const &match,
                     F const &additional_criterion) {
  if (is_singleton_pattern(pattern)) {
    Node pattern_node = get_only(get_nodes(pattern));
    Node graph_matched_node = match.nodeAssignment.at_l(pattern_node);
    if (!additional_criterion(pattern_node, graph_matched_node)) {
      return false;
    }
    for (OpenMultiDiEdge const &e : get_edges(pattern)) {
      MultiDiEdge graph_matched_edge = match.edgeAssignment.at_l(e);

      assert(is_input_edge(e) || is_output_edge(e));
      if (is_input_edge(e)) {
        InputMultiDiEdge input_edge = mpark::get<InputMultiDiEdge>(e);
        if (match.nodeAssignment.at_l(input_edge.dst) !=
                graph_matched_edge.dst ||
            input_edge.dstIdx != graph_matched_edge.dstIdx) {
          return false;
        }
      } else {
        OutputMultiDiEdge output_edge = mpark::get<OutputMultiDiEdge>(e);
        if (match.nodeAssignment.at_l(output_edge.src) !=
                graph_matched_edge.src ||
            output_edge.srcIdx != graph_matched_edge.srcIdx) {
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

  return pattern_matches(*subpatterns.first,
                         graph,
                         submatches.prefix_submatch,
                         additional_criterion) &&
         pattern_matches(*subpatterns.second,
                         graph,
                         submatches.postfix_submatch,
                         additional_criterion);
}

tl::optional<DiGraphPatternMatch>
    get_candidate_singleton_match(IOpenMultiDiGraphView const &pattern,
                                  IMultiDiGraphView const &graph,
                                  Node const &graph_node) {
  assert(is_singleton_pattern(pattern));

  Node pattern_node = get_only(get_nodes(pattern));

  DiGraphPatternMatch match;
  match.nodeAssignment.equate(pattern_node, graph_node);

  auto incoming = get_incoming_edges_by_idx(graph, graph_node);
  auto outgoing = get_outgoing_edges_by_idx(graph, graph_node);
  for (OpenMultiDiEdge const &pattern_edge : get_edges(pattern)) {
    assert(is_input_edge(pattern_edge) || is_output_edge(pattern_edge));
    if (is_input_edge(pattern_edge)) {
      InputMultiDiEdge input_edge = mpark::get<InputMultiDiEdge>(pattern_edge);
      if (!contains_key(incoming, input_edge.dstIdx)) {
        return tl::nullopt;
      }
      match.edgeAssignment.equate(input_edge,
                                  get_only(incoming.at(input_edge.dstIdx)));
    } else {
      OutputMultiDiEdge output_edge =
          mpark::get<OutputMultiDiEdge>(pattern_edge);
      if (!contains_key(outgoing, output_edge.srcIdx)) {
        return tl::nullopt;
      }
      match.edgeAssignment.equate(output_edge,
                                  get_only(outgoing.at(output_edge.srcIdx)));
    }
  }

  return match;
}

tl::optional<DiGraphPatternMatch> unsplit_matches(
    DiGraphPatternMatch const &prefix,
    DiGraphPatternMatch const &postfix,
    bidict<MultiDiEdge, std::pair<OutputMultiDiEdge, InputMultiDiEdge>> const
        &edge_splits) {
  DiGraphPatternMatch result;
  std::unordered_set<OpenMultiDiEdge> handled;
  for (auto const &kv : edge_splits) {
    MultiDiEdge standard_edge = kv.first;
    OutputMultiDiEdge output_edge = kv.second.first;
    InputMultiDiEdge input_edge = kv.second.second;
    handled.insert(output_edge);
    handled.insert(input_edge);

    MultiDiEdge output_graph_edge = prefix.edgeAssignment.at_l(output_edge);
    MultiDiEdge input_graph_edge = postfix.edgeAssignment.at_l(input_edge);
    if (output_graph_edge == input_graph_edge) {
      result.edgeAssignment.equate(standard_edge, output_graph_edge);
    } else {
      return tl::nullopt;
    }
  }

  for (auto const &kv :
       merge_maps(prefix.edgeAssignment, postfix.edgeAssignment)) {
    if (!contains(handled, kv.first)) {
      result.edgeAssignment.equate(kv.first, kv.second);
    }
  }

  result.nodeAssignment =
      merge_maps(prefix.nodeAssignment, postfix.nodeAssignment);

  return result;
}

template <typename F>
std::unordered_set<DiGraphPatternMatch>
    find_pattern_matches(IOpenMultiDiGraphView const &pattern,
                         IMultiDiGraph const &graph,
                         F const &additional_criterion) {
  std::unordered_set<DiGraphPatternMatch> matches;
  if (is_singleton_pattern(pattern)) {
    for (Node const &graph_node : get_nodes(graph)) {
      tl::optional<DiGraphPatternMatch> candidate =
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
    for (DiGraphPatternMatch const &prefix_match : prefix_matches) {
      for (DiGraphPatternMatch const &postfix_match : postfix_matches) {
        tl::optional<DiGraphPatternMatch> unsplit =
            unsplit_matches(prefix_match, postfix_match, edge_splits);
        if (unsplit.has_value()) {
          matches.insert(unsplit.value());
        }
      }
    }
  }

  return matches;
}

} // namespace substitutions
} // namespace FlexFlow
