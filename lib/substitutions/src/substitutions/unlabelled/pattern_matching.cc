#include "substitutions/unlabelled/pattern_matching.h"
#include "substitutions/unlabelled/input_pattern_edge.h"
#include "substitutions/unlabelled/pattern_edge.dtg.h"
#include "substitutions/unlabelled/pattern_node_output.h"
#include "substitutions/unlabelled/pattern_split.h"
#include "substitutions/unlabelled/standard_pattern_edge.h"
#include "substitutions/unlabelled/unlabelled_graph_pattern.h"
#include "utils/bidict/algorithms/left_entries.h"
#include "utils/bidict/algorithms/right_entries.h"
#include "utils/containers/is_subseteq_of.h"
#include "utils/containers/keys.h"
#include "utils/containers/make_counter_func.h"
#include "utils/containers/transform.h"
#include "utils/containers/values.h"
#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_all_open_kwarg_dataflow_edges.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_all_open_kwarg_dataflow_values.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_open_kwarg_dataflow_graph_subgraph.h"
#include "utils/overload.h"
#include <libassert/assert.hpp>
#include <memory>

namespace FlexFlow {

OpenKwargDataflowSubgraphResult<int, TensorSlotName>
    subgraph_matched(OpenKwargDataflowGraphView<int, TensorSlotName> const &g,
                     UnlabelledKwargDataflowGraphPatternMatch const &match) {
  std::set<Node> matched_nodes = right_entries(match.node_assignment);
  return get_open_kwarg_dataflow_graph_subgraph(
      g, matched_nodes, make_counter_func());
}

struct SubgraphConcreteFromPattern {
  SubgraphConcreteFromPattern(
      UnlabelledKwargDataflowGraphPatternMatch const &match,
      bidict<OpenKwargDataflowValue<int, TensorSlotName>,
             KwargDataflowGraphInput<int>> const
          &full_graph_values_to_subgraph_inputs)
      : match(match), full_graph_values_to_subgraph_inputs(
                          full_graph_values_to_subgraph_inputs) {}

  UnlabelledKwargDataflowGraphPatternMatch const &match;
  bidict<OpenKwargDataflowValue<int, TensorSlotName>,
         KwargDataflowGraphInput<int>> const
      &full_graph_values_to_subgraph_inputs;

  Node operator()(PatternNode const &n) const {
    return match.node_assignment.at_l(n);
  }

  OpenKwargDataflowValue<int, TensorSlotName>
      operator()(PatternInput const &i) const {
    OpenKwargDataflowValue<int, TensorSlotName> mapped_input =
        match.input_assignment.at(i);
    if (full_graph_values_to_subgraph_inputs.contains_l(mapped_input)) {
      return OpenKwargDataflowValue<int, TensorSlotName>{
          full_graph_values_to_subgraph_inputs.at_l(mapped_input)};
    } else {
      return mapped_input;
    }
  }

  OpenKwargDataflowEdge<int, TensorSlotName>
      operator()(InputPatternEdge const &e) const {
    return mk_open_kwarg_dataflow_edge_from_src_val_and_dst(
        this->operator()(get_src_input(e)),
        KwargDataflowInput<TensorSlotName>{
            this->operator()(get_dst_node(e)),
            get_dst_slot_name(e),
        });
  }

  KwargDataflowEdge<TensorSlotName>
      operator()(StandardPatternEdge const &e) const {
    return KwargDataflowEdge{
        KwargDataflowOutput{
            this->operator()(get_src_node(e)),
            get_src_slot_name(e),
        },
        KwargDataflowInput{
            this->operator()(get_dst_node(e)),
            get_dst_slot_name(e),
        },
    };
  }

  OpenKwargDataflowEdge<int, TensorSlotName>
      operator()(PatternEdge const &pattern_e) const {
    return pattern_e.visit<OpenKwargDataflowEdge<int, TensorSlotName>>(
        [&](auto const &e) {
          return OpenKwargDataflowEdge<int, TensorSlotName>{
              this->operator()(e),
          };
        });
  }

  OpenKwargDataflowValue<int, TensorSlotName>
      operator()(PatternValue const &pattern_v) const {
    return pattern_v.visit<OpenKwargDataflowValue<int, TensorSlotName>>(
        [&](auto const &v) {
          return OpenKwargDataflowValue<int, TensorSlotName>{
              this->operator()(v)};
        });
  }

  KwargDataflowOutput<TensorSlotName>
      operator()(PatternNodeOutput const &o) const {
    return KwargDataflowOutput<TensorSlotName>{
        this->operator()(get_src_node(o)),
        get_slot_name(o),
    };
  }
};

bool pattern_matches_subgraph_under(
    UnlabelledGraphPattern const &pattern,
    OpenKwargDataflowGraphView<int, TensorSlotName> const &subgraph,
    bidict<OpenKwargDataflowValue<int, TensorSlotName>,
           KwargDataflowGraphInput<int>> const
        &full_graph_values_to_subgraph_inputs,
    UnlabelledKwargDataflowGraphPatternMatch const &match,
    MatchAdditionalCriterion const &additional_criterion) {
  SubgraphConcreteFromPattern concrete_from_pattern{
      match, full_graph_values_to_subgraph_inputs};

  std::set<Node> concrete_nodes = get_nodes(subgraph);
  std::set<Node> concrete_nodes_from_match =
      transform(get_pattern_nodes(pattern), concrete_from_pattern);

  if (concrete_nodes != concrete_nodes_from_match) {
    return false;
  }

  for (PatternNode const &pattern_node : get_pattern_nodes(pattern)) {
    if (!additional_criterion.node_criterion(
            pattern_node, concrete_from_pattern(pattern_node))) {
      return false;
    }
  }

  std::set<OpenKwargDataflowEdge<int, TensorSlotName>> concrete_edges =
      get_all_open_kwarg_dataflow_edges(subgraph);
  std::set<OpenKwargDataflowEdge<int, TensorSlotName>>
      concrete_edge_from_match =
          transform(get_pattern_edges(pattern),
                    [&](PatternEdge const &e)
                        -> OpenKwargDataflowEdge<int, TensorSlotName> {
                      return concrete_from_pattern(e);
                    });

  if (concrete_edges != concrete_edge_from_match) {
    return false;
  }

  std::set<OpenKwargDataflowValue<int, TensorSlotName>> concrete_values =
      get_all_open_kwarg_dataflow_values(subgraph);
  std::set<OpenKwargDataflowValue<int, TensorSlotName>>
      concrete_values_from_match =
          transform(get_pattern_values(pattern),
                    [&](PatternValue const &v)
                        -> OpenKwargDataflowValue<int, TensorSlotName> {
                      return concrete_from_pattern(v);
                    });

  if (concrete_values != concrete_values_from_match) {
    return false;
  }

  for (PatternValue const &pattern_value : get_pattern_values(pattern)) {
    if (!additional_criterion.value_criterion(
            pattern_value, concrete_from_pattern(pattern_value))) {
      return false;
    }
  }

  return true;
}

bool unlabelled_pattern_does_match(
    UnlabelledGraphPattern const &pattern,
    OpenKwargDataflowGraphView<int, TensorSlotName> const &graph,
    UnlabelledKwargDataflowGraphPatternMatch const &match,
    MatchAdditionalCriterion const &additional_criterion) {
  return unlabelled_pattern_does_match(pattern,
                                       graph,
                                       match,
                                       additional_criterion,
                                       std::set<StandardPatternEdge>{});
}

bool unlabelled_pattern_does_match(
    UnlabelledGraphPattern const &pattern,
    OpenKwargDataflowGraphView<int, TensorSlotName> const &graph,
    UnlabelledKwargDataflowGraphPatternMatch const &match,
    MatchAdditionalCriterion const &additional_criterion,
    std::set<StandardPatternEdge> const &hint_edges) {
  std::set<OpenKwargDataflowValue<int, TensorSlotName>>
      matched_by_pattern_inputs = set_of(values(match.input_assignment));

  ASSERT(left_entries(match.node_assignment) == get_pattern_nodes(pattern));
  ASSERT(
      is_subseteq_of(right_entries(match.node_assignment), get_nodes(graph)));
  ASSERT(keys(match.input_assignment) == get_pattern_inputs(pattern));
  ASSERT(is_subseteq_of(matched_by_pattern_inputs,
                        get_all_open_kwarg_dataflow_values(graph)));

  std::set<Node> subgraph_nodes = match.node_assignment.right_values();

  // TODO(@lockshaw)(#pr): maybe clean this up
  OpenKwargDataflowEdgeQuery<int, TensorSlotName>
      subgraph_interior_edges_query =
          OpenKwargDataflowEdgeQuery<int, TensorSlotName>{
              KwargDataflowInputEdgeQuery<int, TensorSlotName>{
                  /*srcs=*/query_set<int>::match_none(),
                  /*dst_nodes=*/query_set<Node>::match_none(),
                  /*dst_slots=*/query_set<TensorSlotName>::match_none(),
              },
              KwargDataflowEdgeQuery<TensorSlotName>{
                  /*srcs=*/query_set<Node>::match_values_in(subgraph_nodes),
                  /*src_slots=*/query_set<TensorSlotName>::matchall(),
                  /*dsts=*/query_set<Node>::match_values_in(subgraph_nodes),
                  /*dst_slots=*/query_set<TensorSlotName>::matchall(),
              },
          };

  std::set<KwargDataflowEdge<TensorSlotName>> subgraph_interior_edges =
      transform(graph.query_edges(subgraph_interior_edges_query),
                [](OpenKwargDataflowEdge<int, TensorSlotName> const &e) {
                  return e.require_internal_edge();
                });

  for (StandardPatternEdge const &e : hint_edges) {
    KwargDataflowEdge<TensorSlotName> concrete_edge = e.raw_edge;
    concrete_edge.src.node =
        match.node_assignment.at_l(PatternNode{concrete_edge.src.node});
    concrete_edge.dst.node =
        match.node_assignment.at_l(PatternNode{concrete_edge.dst.node});

    if (!contains(subgraph_interior_edges, concrete_edge)) {
      return false;
    }
  }

  OpenKwargDataflowSubgraphResult<int, TensorSlotName> subgraph_result =
      subgraph_matched(graph, match);
  OpenKwargDataflowGraphView<int, TensorSlotName> matched_subgraph =
      subgraph_result.graph;

  std::set<OpenKwargDataflowValue<int, TensorSlotName>>
      full_values_split_by_subgraph =
          left_entries(subgraph_result.full_graph_values_to_subgraph_inputs);

  ASSERT(right_entries(match.node_assignment) == get_nodes(matched_subgraph));
  ASSERT(is_subseteq_of(full_values_split_by_subgraph,
                        get_all_open_kwarg_dataflow_values(graph)),
         full_values_split_by_subgraph,
         get_all_open_kwarg_dataflow_values(graph));

  MatchAdditionalCriterion through_subgraph_operation =
      MatchAdditionalCriterion{
          additional_criterion.node_criterion,
          [&](PatternValue const &pv,
              OpenKwargDataflowValue<int, TensorSlotName> const &v) {
            return v.visit<bool>(overload{
                [&](KwargDataflowOutput<TensorSlotName> const &) {
                  return additional_criterion.value_criterion(pv, v);
                },
                [&](KwargDataflowGraphInput<int> const &subgraph_input) {
                  OpenKwargDataflowValue<int, TensorSlotName> full_graph_value =
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
