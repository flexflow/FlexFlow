#include "substitutions/unlabelled/unlabelled_graph_pattern.h"
#include "substitutions/unlabelled/pattern_edge.h"
#include "substitutions/unlabelled/pattern_value.h"
#include "utils/containers/transform.h"
#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_topological_ordering.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_inputs.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_edges.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_open_dataflow_values.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_subgraph.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_subgraph_inputs.h"

namespace FlexFlow {

size_t num_nodes(UnlabelledGraphPattern const &p) {
  return num_nodes(p.raw_graph);
}

bool is_singleton_pattern(UnlabelledGraphPattern const &pattern) {
  return num_nodes(pattern) == 1;
}

std::unordered_set<PatternNode> get_nodes(UnlabelledGraphPattern const &p) {
  return transform(get_nodes(p.raw_graph),
                   [](Node const &n) { return PatternNode{n}; });
}

std::unordered_set<PatternValue> get_values(UnlabelledGraphPattern const &p) {
  return transform(get_open_dataflow_values(p.raw_graph),
                   pattern_value_from_raw_open_dataflow_value);
}

std::unordered_set<PatternInput> get_inputs(UnlabelledGraphPattern const &p) {
  return transform(get_inputs(p.raw_graph),
                   [](DataflowGraphInput const &i) { return PatternInput{i}; });
}

std::unordered_set<PatternEdge> get_edges(UnlabelledGraphPattern const &p) {
  return transform(get_edges(p.raw_graph),
                   pattern_edge_from_raw_open_dataflow_edge);
}

std::vector<PatternNode>
    get_topological_ordering(UnlabelledGraphPattern const &p) {
  return transform(get_topological_ordering(p.raw_graph),
                   [](Node const &n) { return PatternNode{n}; });
}

std::vector<PatternValue>
    get_inputs_to_pattern_node(UnlabelledGraphPattern const &p,
                               PatternNode const &n) {
  return transform(get_inputs(p.raw_graph, n.raw_node),
                   pattern_value_from_raw_open_dataflow_value);
}

std::vector<PatternValue>
    get_outputs_from_pattern_node(UnlabelledGraphPattern const &p,
                                  PatternNode const &n) {
  return transform(
      get_outputs(p.raw_graph, n.raw_node), [](DataflowOutput const &o) {
        return pattern_value_from_raw_open_dataflow_value(OpenDataflowValue{o});
      });
}

UnlabelledGraphPatternSubgraphResult
    get_subgraph(UnlabelledGraphPattern const &p,
                 std::unordered_set<PatternNode> const &n) {
  OpenDataflowSubgraphResult raw_result = get_subgraph(
      p.raw_graph,
      transform(n, [](PatternNode const &pn) { return pn.raw_node; }));
  bidict<PatternValue, PatternInput> full_pattern_values_to_subpattern_inputs =
      transform(raw_result.full_graph_values_to_subgraph_inputs,
                [](OpenDataflowValue const &v, DataflowGraphInput const &i) {
                  return std::make_pair(
                      pattern_value_from_raw_open_dataflow_value(v),
                      PatternInput{i});
                });
  return UnlabelledGraphPatternSubgraphResult{
      UnlabelledGraphPattern{raw_result.graph},
      full_pattern_values_to_subpattern_inputs,
  };
}

} // namespace FlexFlow
