#include "substitutions/unlabelled/unlabelled_graph_pattern.h"
#include "substitutions/unlabelled/pattern_edge.h"
#include "substitutions/unlabelled/pattern_value.h"
#include "utils/bidict/algorithms/transform.h"
#include "utils/containers/make_counter_func.h"
#include "utils/containers/transform.h"
#include "utils/graph/digraph/algorithms/get_topological_ordering.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_outgoing_kwarg_dataflow_outputs_for_node.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_all_kwarg_dataflow_graph_inputs.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_all_open_kwarg_dataflow_edges.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_all_open_kwarg_dataflow_values.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_incoming_open_kwarg_dataflow_values_for_node.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_open_kwarg_dataflow_graph_subgraph.h"

namespace FlexFlow {

size_t num_nodes(UnlabelledGraphPattern const &p) {
  return num_nodes(p.raw_graph);
}

bool is_singleton_pattern(UnlabelledGraphPattern const &pattern) {
  return num_nodes(pattern) == 1;
}

std::set<PatternNode> get_pattern_nodes(UnlabelledGraphPattern const &p) {
  return transform(get_nodes(p.raw_graph),
                   [](Node const &n) { return PatternNode{n}; });
}

std::set<PatternValue> get_pattern_values(UnlabelledGraphPattern const &p) {
  return transform(get_all_open_kwarg_dataflow_values(p.raw_graph),
                   pattern_value_from_raw_open_kwarg_dataflow_value);
}

std::set<PatternInput> get_pattern_inputs(UnlabelledGraphPattern const &p) {
  return transform(
      get_all_kwarg_dataflow_graph_inputs(p.raw_graph),
      [](KwargDataflowGraphInput<int> const &i) { return PatternInput{i}; });
}

std::set<StandardPatternEdge>
    get_pattern_edges_across_topological_split(UnlabelledGraphPattern const &p,
                                               PatternSplit const &s) {
  std::set<Node> src_nodes =
      transform(s.first, [](PatternNode const &n) { return n.raw_node; });
  std::set<Node> dst_nodes =
      transform(s.second, [](PatternNode const &n) { return n.raw_node; });

  OpenKwargDataflowEdgeQuery q = OpenKwargDataflowEdgeQuery{
      /*input_edge_query=*/kwarg_dataflow_input_edge_query_none<
          int,
          TensorSlotName>(),
      /*standard_edge_query=*/
      KwargDataflowEdgeQuery<TensorSlotName>{
          /*src_nodes=*/query_set<Node>::match_values_in(src_nodes),
          /*src_slots=*/query_set<TensorSlotName>::matchall(),
          /*dst_nodes=*/query_set<Node>::match_values_in(dst_nodes),
          /*dst_slots=*/query_set<TensorSlotName>::matchall(),
      },
  };

  /* TODO(@lockshaw)(#pr):
   * debug assert that the edges are in fact one directional
   */

  return transform(p.raw_graph.query_edges(q),
                   [](OpenKwargDataflowEdge<int, TensorSlotName> const &e) {
                     return StandardPatternEdge{e.require_internal_edge()};
                   });
}

std::set<PatternEdge> get_pattern_edges(UnlabelledGraphPattern const &p) {
  return transform(get_all_open_kwarg_dataflow_edges(p.raw_graph),
                   pattern_edge_from_raw_open_dataflow_edge);
}

std::vector<PatternNode>
    get_topological_ordering(UnlabelledGraphPattern const &p) {
  return transform(get_topological_ordering(p.raw_graph),
                   [](Node const &n) { return PatternNode{n}; });
}

std::map<TensorSlotName, PatternValue>
    get_inputs_to_pattern_node(UnlabelledGraphPattern const &p,
                               PatternNode const &n) {
  return map_values(
      get_incoming_open_kwarg_dataflow_values_for_node(p.raw_graph, n.raw_node),
      [](OpenKwargDataflowValue<int, TensorSlotName> const &v) -> PatternValue {
        return pattern_value_from_raw_open_kwarg_dataflow_value(v);
      });
}

std::map<TensorSlotName, PatternValue>
    get_outputs_from_pattern_node(UnlabelledGraphPattern const &p,
                                  PatternNode const &n) {
  return map_values(
      get_outgoing_kwarg_dataflow_outputs_for_node(p.raw_graph, n.raw_node),
      [](KwargDataflowOutput<TensorSlotName> const &o) {
        return pattern_value_from_raw_open_kwarg_dataflow_value(
            OpenKwargDataflowValue<int, TensorSlotName>{o});
      });
}

UnlabelledGraphPatternSubgraphResult
    get_pattern_subgraph(UnlabelledGraphPattern const &p,
                         std::set<PatternNode> const &n) {
  OpenKwargDataflowSubgraphResult raw_result =
      get_open_kwarg_dataflow_graph_subgraph(
          p.raw_graph,
          transform(n, [](PatternNode const &pn) { return pn.raw_node; }),
          make_counter_func());
  bidict<PatternValue, PatternInput> full_pattern_values_to_subpattern_inputs =
      transform(raw_result.full_graph_values_to_subgraph_inputs,
                [](OpenKwargDataflowValue<int, TensorSlotName> const &v,
                   KwargDataflowGraphInput<int> const &i) {
                  return std::make_pair(
                      pattern_value_from_raw_open_kwarg_dataflow_value(v),
                      PatternInput{i});
                });
  return UnlabelledGraphPatternSubgraphResult{
      UnlabelledGraphPattern{raw_result.graph},
      full_pattern_values_to_subpattern_inputs,
  };
}

} // namespace FlexFlow
