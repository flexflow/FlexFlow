#include "substitutions/unlabelled/find_pattern_matches.h"
#include "substitutions/unlabelled/downward_open_pattern_edge.h"
#include "substitutions/unlabelled/multidigraph_pattern_match.h"
#include "substitutions/unlabelled/unlabelled_graph_pattern.h"
#include "substitutions/unlabelled/upward_open_pattern_edge.h"
#include "utils/containers.h"

namespace FlexFlow {

static std::vector<UpwardOpenPatternEdge>
    sorted_by_dst_idx(std::unordered_set<UpwardOpenPatternEdge> const &in) {
  return sorted_by(
      in, compare_by<UpwardOpenPatternEdge>([](UpwardOpenPatternEdge const &e) {
        return get_dst_idx(e);
      }));
}

static std::vector<DownwardOpenPatternEdge>
    sorted_by_src_idx(std::unordered_set<DownwardOpenPatternEdge> const &in) {
  return sorted_by(
      in,
      compare_by<DownwardOpenPatternEdge>(
          [](DownwardOpenPatternEdge const &e) { return get_src_idx(e); }));
}

static std::vector<UpwardOpenMultiDiEdge>
    sorted_by_dst_idx(std::unordered_set<UpwardOpenMultiDiEdge> const &in) {
  return sorted_by(
      in, compare_by<UpwardOpenPatternEdge>([](UpwardOpenPatternEdge const &e) {
        return get_dst_idx(e);
      }));
}

static std::vector<DownwardOpenMultiDiEdge>
    sorted_by_src_idx(std::unordered_set<DownwardOpenMultiDiEdge> const &in) {
  return sorted_by(
      in,
      compare_by<DownwardOpenMultiDiEdge>(
          [](DownwardOpenMultiDiEdge const &e) { return get_src_idx(e); }));
}

static std::optional<MultiDiGraphPatternMatch>
    get_candidate_singleton_match(UnlabelledGraphPattern const &pattern,
                                  OpenMultiDiGraphView const &graph,
                                  Node const &graph_node) {
  assert(is_singleton_pattern(pattern));

  PatternNode pattern_node = get_only(get_nodes(pattern));

  MultiDiGraphPatternMatch match = empty_multidigraph_pattern_match();
  match.node_assignment.equate(pattern_node, graph_node);

  std::unordered_set<UpwardOpenMultiDiEdge> incoming =
      get_incoming_edges(graph, graph_node);
  std::unordered_set<DownwardOpenMultiDiEdge> outgoing =
      get_outgoing_edges(graph, graph_node);

  std::unordered_set<UpwardOpenPatternEdge> pattern_incoming =
      get_incoming_edges(pattern, pattern_node);
  std::unordered_set<DownwardOpenPatternEdge> pattern_outgoing =
      get_outgoing_edges(pattern, pattern_node);

  if (!pattern_incoming.empty() && pattern_incoming.size() != incoming.size()) {
    return std::nullopt;
  }

  if (!pattern_outgoing.empty() && pattern_outgoing.size() != outgoing.size()) {
    return std::nullopt;
  }

  std::vector<UpwardOpenMultiDiEdge> incoming_ordered =
      sorted_by_dst_idx(incoming);
  std::vector<DownwardOpenMultiDiEdge> outgoing_ordered =
      sorted_by_src_idx(outgoing);

  std::vector<UpwardOpenPatternEdge> pattern_incoming_ordered =
      sorted_by_dst_idx(pattern_incoming);
  std::vector<DownwardOpenPatternEdge> pattern_outgoing_ordered =
      sorted_by_src_idx(pattern_outgoing);

  if (pattern_incoming.size() > 0) {
    std::unordered_map<NodePort, NodePort> node_port_mapping;
    for (int i = 0; i < incoming_ordered.size(); ++i) {
      UpwardOpenMultiDiEdge graph_edge = incoming_ordered[i];
      UpwardOpenPatternEdge pattern_edge = pattern_incoming_ordered[i];
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

  if (pattern_outgoing.size() > 0) {
    std::unordered_map<NodePort, NodePort> node_port_mapping;
    for (int i = 0; i < outgoing_ordered.size(); ++i) {
      DownwardOpenMultiDiEdge graph_edge = outgoing_ordered[i],
                              DownwardOpenPatternEdge pattern_edge =
                                  pattern_outgoing_ordered[i];

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

std::vector<MultiDiGraphPatternMatch>
    find_pattern_matches(UnlabelledGraphPattern const &pattern,
                         OpenMultiDiGraphView const &graph,
                         MatchAdditionalCriterion const &additional_criterion) {
  std::vector<MultiDiGraphPatternMatch> matches;
  if (is_singleton_pattern(pattern)) {
    for (Node const &graph_node : get_nodes(graph)) {
      std::optional<MultiDiGraphPatternMatch> candidate =
          get_candidate_singleton_match(pattern, graph, graph_node);
      if (candidate.has_value() &&
          pattern_does_match(
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
