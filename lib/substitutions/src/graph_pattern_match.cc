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
  return {get_subgraph<OpenMultiDiSubgraphView>(pattern, split.first),
          get_subgraph<OpenMultiDiSubgraphView>(pattern, split.second)};
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

  std::function<void(OpenMultiDiEdge const &, OpenMultiDiEdge const &)>
      handle_edge = [&](OpenMultiDiEdge const &pattern_edge,
                        OpenMultiDiEdge const &graph_edge) -> void {
    auto edge_nodes = get_nodes(pattern_edge);
    if (is_subseteq_of(edge_nodes, prefix)) {
      result.prefix_submatch.edge_assignment.equate(pattern_edge, graph_edge);
    } else if (is_subseteq_of(edge_nodes, postfix)) {
      result.postfix_submatch.edge_assignment.equate(pattern_edge, graph_edge);
    } else {
      assert(is_standard_edge(pattern_edge));
      assert(is_standard_edge(graph_edge));
      auto standard_edge = std::get<MultiDiEdge>(pattern_edge);
      auto divided = edge_splits.at_l(standard_edge);
      auto divided_graph_edge = split_edge(get<MultiDiEdge>(graph_edge));
      handle_edge(divided.first, divided_graph_edge.first);
      handle_edge(divided.second, divided_graph_edge.second);
    }
  };

  for (auto const &kv : match.edge_assignment) {
    OpenMultiDiEdge pattern_edge = kv.first;
    OpenMultiDiEdge graph_edge = match.edge_assignment.at_l(pattern_edge);
    handle_edge(pattern_edge, graph_edge);
  }

  return result;
}

bool is_singleton_pattern(OpenMultiDiGraphView const &pattern) {
  return num_nodes(pattern) == 1;
}

bool pattern_matches(OpenMultiDiGraphView const &pattern,
                     OpenMultiDiGraphView const &graph,
                     MultiDiGraphPatternMatch const &match,
                     MatchAdditionalCriterion const &additional_criterion) {
  if (is_singleton_pattern(pattern)) {
    Node pattern_node = get_only(get_nodes(pattern));
    Node graph_matched_node = match.node_assignment.at_l(pattern_node);
    if (!additional_criterion.node_criterion(pattern_node,
                                             graph_matched_node)) {
      return false;
    }
    for (OpenMultiDiEdge const &e : get_edges(pattern)) {
      OpenMultiDiEdge graph_matched_edge = match.edge_assignment.at_l(e);

      assert(is_input_edge(e) || is_output_edge(e));
      if (is_input_edge(e)) {
        if (is_output_edge(graph_matched_edge)) {
          return false;
        }
        UpwardOpenMultiDiEdge matched_edge =
            narrow<UpwardOpenMultiDiEdge>(graph_matched_edge).value();
        InputMultiDiEdge input_edge = std::get<InputMultiDiEdge>(e);
        if (match.node_assignment.at_l(input_edge.dst) !=
            get_dst_node(matched_edge)) {
          return false;
        }
      } else {
        if (is_input_edge(graph_matched_edge)) {
          return false;
        }
        DownwardOpenMultiDiEdge matched_edge =
            narrow<DownwardOpenMultiDiEdge>(graph_matched_edge).value();
        OutputMultiDiEdge output_edge = std::get<OutputMultiDiEdge>(e);
        if (match.node_assignment.at_l(output_edge.src) !=
            get_src_node(matched_edge)) {
          return false;
        }
      }

      if (!additional_criterion.edge_criterion(e, graph_matched_edge)) {
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

template <typename T>
bool dst_compare(T const &lhs, T const &rhs) {
  return get_dst_idx(lhs) < get_dst_idx(rhs);
}

template <typename T>
bool src_compare(T const &lhs, T const &rhs) {
  return get_src_idx(lhs) < get_src_idx(rhs);
}

std::optional<MultiDiGraphPatternMatch>
    get_candidate_singleton_match(OpenMultiDiGraphView const &pattern,
                                  OpenMultiDiGraphView const &graph,
                                  Node const &graph_node) {
  assert(is_singleton_pattern(pattern));

  Node pattern_node = get_only(get_nodes(pattern));

  MultiDiGraphPatternMatch match;
  match.node_assignment.equate(pattern_node, graph_node);

  std::unordered_set<UpwardOpenMultiDiEdge> incoming =
      get_incoming_edges(graph, graph_node);
  std::unordered_set<DownwardOpenMultiDiEdge> outgoing =
      get_outgoing_edges(graph, graph_node);

  std::unordered_set<UpwardOpenMultiDiEdge> pattern_incoming =
      get_incoming_edges(pattern, pattern_node);
  std::unordered_set<DownwardOpenMultiDiEdge> pattern_outgoing =
      get_outgoing_edges(pattern, pattern_node);

  if (!pattern_incoming.empty() && pattern_incoming.size() != incoming.size()) {
    return std::nullopt;
  }

  if (!pattern_outgoing.empty() && pattern_outgoing.size() != outgoing.size()) {
    return std::nullopt;
  }

  std::vector<UpwardOpenMultiDiEdge> incoming_ordered =
      sorted_by(incoming, dst_compare<UpwardOpenMultiDiEdge>);
  std::vector<DownwardOpenMultiDiEdge> outgoing_ordered =
      sorted_by(outgoing, src_compare<DownwardOpenMultiDiEdge>);

  std::vector<UpwardOpenMultiDiEdge> pattern_incoming_ordered =
      sorted_by(pattern_incoming, dst_compare<UpwardOpenMultiDiEdge>);
  std::vector<DownwardOpenMultiDiEdge> pattern_outgoing_ordered =
      sorted_by(pattern_outgoing, src_compare<DownwardOpenMultiDiEdge>);

  if (pattern_incoming.size()) {
    std::unordered_map<NodePort, NodePort> node_port_mapping;
    for (int i = 0; i < incoming_ordered.size(); ++i) {
      UpwardOpenMultiDiEdge graph_edge = incoming_ordered[i],
                            pattern_edge = pattern_incoming_ordered[i];
      NodePort graph_port = get_dst_idx(graph_edge),
               pattern_port = get_dst_idx(pattern_edge);
      if (!contains_key(node_port_mapping, graph_port)) {
        node_port_mapping.emplace(graph_port, pattern_port);
      } else {
        if (pattern_port != node_port_mapping.at(graph_port)) {
          return std::nullopt;
        }
      }
      match.edge_assignment.equate(widen<OpenMultiDiEdge>(pattern_edge),
                                   widen<OpenMultiDiEdge>(graph_edge));
    }
  }

  if (pattern_outgoing.size()) {
    std::unordered_map<NodePort, NodePort> node_port_mapping;
    for (int i = 0; i < outgoing_ordered.size(); ++i) {
      DownwardOpenMultiDiEdge graph_edge = outgoing_ordered[i],
                              pattern_edge = pattern_outgoing_ordered[i];
      NodePort graph_port = get_src_idx(graph_edge),
               pattern_port = get_src_idx(pattern_edge);
      if (!contains_key(node_port_mapping, graph_port)) {
        node_port_mapping.insert({graph_port, pattern_port});
      } else {
        if (pattern_port != node_port_mapping.at(graph_port)) {
          return std::nullopt;
        }
      }
      match.edge_assignment.equate(widen<OpenMultiDiEdge>(pattern_edge),
                                   widen<OpenMultiDiEdge>(graph_edge));
    }
  }

  return match;
}

std::optional<MultiDiGraphPatternMatch> unsplit_matches(
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
      return std::nullopt;
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

std::vector<MultiDiGraphPatternMatch>
    find_pattern_matches(OpenMultiDiGraphView const &pattern,
                         OpenMultiDiGraphView const &graph,
                         MatchAdditionalCriterion const &additional_criterion) {
  std::vector<MultiDiGraphPatternMatch> matches;
  if (is_singleton_pattern(pattern)) {
    for (Node const &graph_node : get_nodes(graph)) {
      std::optional<MultiDiGraphPatternMatch> candidate =
          get_candidate_singleton_match(pattern, graph, graph_node);
      if (candidate.has_value() &&
          pattern_matches(
              pattern, graph, candidate.value(), additional_criterion)) {
        matches.push_back(candidate.value());
      }
    }
  } else {
    GraphSplit split = split_pattern(pattern);
    auto subpatterns = apply_split(pattern, split);
    auto prefix_matches =
        find_pattern_matches(subpatterns.first, graph, additional_criterion);
    auto postfix_matches =
        find_pattern_matches(subpatterns.second, graph, additional_criterion);
    auto edge_splits = get_edge_splits(pattern, split);
    for (MultiDiGraphPatternMatch const &prefix_match : prefix_matches) {
      for (MultiDiGraphPatternMatch const &postfix_match : postfix_matches) {
        std::optional<MultiDiGraphPatternMatch> unsplit =
            unsplit_matches(prefix_match, postfix_match, edge_splits);
        if (unsplit.has_value()) {
          matches.push_back(unsplit.value());
        }
      }
    }
  }

  return matches;
}

} // namespace FlexFlow
