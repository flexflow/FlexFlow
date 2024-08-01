#include "substitutions/unlabelled/pattern_matching.h"
#include "substitutions/unlabelled/input_pattern_edge.h"
#include "substitutions/unlabelled/pattern_edge.dtg.h"
#include "substitutions/unlabelled/pattern_node_output.h"
#include "substitutions/unlabelled/pattern_split.h"
#include "substitutions/unlabelled/standard_pattern_edge.h"
#include "substitutions/unlabelled/unlabelled_graph_pattern.h"
#include "utils/containers/keys.h"
#include "utils/containers/transform.h"
#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/open_dataflow_graph/algorithms.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_open_dataflow_values.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_subgraph.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_subgraph_inputs.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_edge.dtg.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_edge.h"
#include "utils/overload.h"
#include <memory>

namespace FlexFlow {

OpenDataflowSubgraphResult
    subgraph_matched(OpenDataflowGraphView const &g,
                     UnlabelledDataflowGraphPatternMatch const &match) {
  std::unordered_set<Node> matched_nodes =
      keys(match.node_assignment.reversed());
  return get_subgraph(g, matched_nodes);
}

struct SubgraphConcreteFromPattern {
  SubgraphConcreteFromPattern(
      UnlabelledDataflowGraphPatternMatch const &match,
      bidict<OpenDataflowValue, DataflowGraphInput> const
          &full_graph_values_to_subgraph_inputs)
      : match(match), full_graph_values_to_subgraph_inputs(
                          full_graph_values_to_subgraph_inputs) {}

  UnlabelledDataflowGraphPatternMatch const &match;
  bidict<OpenDataflowValue, DataflowGraphInput> const
      &full_graph_values_to_subgraph_inputs;

  Node operator()(PatternNode const &n) const {
    return match.node_assignment.at_l(n);
  }

  OpenDataflowValue operator()(PatternInput const &i) const {
    return OpenDataflowValue{full_graph_values_to_subgraph_inputs.at_l(
        match.input_assignment.at(i))};
  }

  OpenDataflowEdge operator()(InputPatternEdge const &e) const {
    return open_dataflow_edge_from_src_and_dst(
        this->operator()(get_src_input(e)),
        DataflowInput{
            this->operator()(get_dst_node(e)),
            get_dst_idx(e),
        });
  }

  DataflowEdge operator()(StandardPatternEdge const &e) const {
    return DataflowEdge{
        DataflowOutput{
            this->operator()(get_src_node(e)),
            get_src_idx(e),
        },
        DataflowInput{
            this->operator()(get_dst_node(e)),
            get_dst_idx(e),
        },
    };
  }

  OpenDataflowEdge operator()(PatternEdge const &pattern_e) const {
    return pattern_e.visit<OpenDataflowEdge>(
        [&](auto const &e) { return OpenDataflowEdge{this->operator()(e)}; });
  }

  OpenDataflowValue operator()(PatternValue const &pattern_v) const {
    return pattern_v.visit<OpenDataflowValue>(
        [&](auto const &v) { return OpenDataflowValue{this->operator()(v)}; });
  }

  DataflowOutput operator()(PatternNodeOutput const &o) const {
    return DataflowOutput{
        this->operator()(get_src_node(o)),
        get_idx(o),
    };
  }
};

bool pattern_matches_subgraph_under(
    UnlabelledGraphPattern const &pattern,
    OpenDataflowGraphView const &subgraph,
    bidict<OpenDataflowValue, DataflowGraphInput> const
        &full_graph_values_to_subgraph_inputs,
    UnlabelledDataflowGraphPatternMatch const &match,
    MatchAdditionalCriterion const &additional_criterion) {
  SubgraphConcreteFromPattern concrete_from_pattern{
      match, full_graph_values_to_subgraph_inputs};

  std::unordered_set<Node> concrete_nodes = get_nodes(subgraph);
  std::unordered_set<Node> concrete_nodes_from_match =
      transform(get_nodes(pattern), concrete_from_pattern);

  if (concrete_nodes != concrete_nodes_from_match) {
    return false;
  }

  for (PatternNode const &pattern_node : get_nodes(pattern)) {
    if (!additional_criterion.node_criterion(
            pattern_node, concrete_from_pattern(pattern_node))) {
      return false;
    }
  }

  std::unordered_set<OpenDataflowEdge> concrete_edges = get_edges(subgraph);
  std::unordered_set<OpenDataflowEdge> concrete_edge_from_match =
      transform(get_edges(pattern), concrete_from_pattern);

  if (concrete_edges != concrete_edge_from_match) {
    return false;
  }

  std::unordered_set<OpenDataflowValue> concrete_values =
      get_open_dataflow_values(subgraph);
  std::unordered_set<OpenDataflowValue> concrete_values_from_match =
      transform(get_values(pattern), concrete_from_pattern);

  if (concrete_values != concrete_values_from_match) {
    return false;
  }

  for (PatternValue const &pattern_value : get_values(pattern)) {
    if (!additional_criterion.value_criterion(
            pattern_value, concrete_from_pattern(pattern_value))) {
      return false;
    }
  }

  return true;
}

bool unlabelled_pattern_does_match(
    UnlabelledGraphPattern const &pattern,
    OpenDataflowGraphView const &graph,
    UnlabelledDataflowGraphPatternMatch const &match,
    MatchAdditionalCriterion const &additional_criterion) {

  OpenDataflowSubgraphResult subgraph_result = subgraph_matched(graph, match);
  OpenDataflowGraphView matched_subgraph = subgraph_result.graph;

  assert(keys(match.node_assignment) == get_nodes(pattern));
  assert(keys(match.node_assignment.reversed()) == get_nodes(matched_subgraph));

  MatchAdditionalCriterion through_subgraph_operation =
      MatchAdditionalCriterion{
          additional_criterion.node_criterion,
          [&](PatternValue const &pv, OpenDataflowValue const &v) {
            return v.visit<bool>(overload{
                [&](DataflowOutput const &) {
                  return additional_criterion.value_criterion(pv, v);
                },
                [&](DataflowGraphInput const &subgraph_input) {
                  OpenDataflowValue full_graph_value =
                      subgraph_result.full_graph_values_to_subgraph_inputs.at_r(
                          subgraph_input);
                  return additional_criterion.value_criterion(pv,
                                                              full_graph_value);
                }});
          },
      };

  return pattern_matches_subgraph_under(
      pattern,
      matched_subgraph,
      subgraph_result.full_graph_values_to_subgraph_inputs,
      match,
      through_subgraph_operation);
}

} // namespace FlexFlow
