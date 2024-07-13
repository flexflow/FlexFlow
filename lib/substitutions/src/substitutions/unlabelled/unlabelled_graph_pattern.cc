#include "substitutions/unlabelled/unlabelled_graph_pattern.h"
#include "substitutions/unlabelled/pattern_edge.h"
#include "substitutions/unlabelled/pattern_value.h"
#include "utils/containers.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/open_dataflow_graph/algorithms.h"
#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/digraph/algorithms.h"
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
  return transform(get_open_dataflow_values(p.raw_graph), pattern_value_from_raw_open_dataflow_value);
}

std::unordered_set<PatternInput> get_inputs(UnlabelledGraphPattern const &p) {
  return transform(get_inputs(p.raw_graph), [](DataflowGraphInput const &i) { return PatternInput{i}; });
}

std::unordered_set<PatternEdge> get_edges(UnlabelledGraphPattern const &p) {
  return transform(get_edges(p.raw_graph), pattern_edge_from_raw_open_dataflow_edge);
}

std::vector<PatternNode> get_topological_ordering(UnlabelledGraphPattern const &p) {
  return transform(get_topological_ordering(p.raw_graph),
                   [](Node const &n) { return PatternNode{n}; });
}

std::vector<PatternValue>
    get_inputs_to_pattern_node(UnlabelledGraphPattern const &p, PatternNode const &n) {
  return transform(get_inputs(p.raw_graph, n.raw_node), pattern_value_from_raw_open_dataflow_value);
}

std::vector<PatternValue>
    get_outputs_from_pattern_node(UnlabelledGraphPattern const &p, PatternNode const &n) {
  return transform(get_outputs(p.raw_graph, n.raw_node), 
                   [](DataflowOutput const &o) { return pattern_value_from_raw_open_dataflow_value(OpenDataflowValue{o}); });
}

UnlabelledGraphPattern get_subgraph(UnlabelledGraphPattern const &p,
                                    std::unordered_set<PatternNode> const &n) {
  OpenDataflowGraphView raw_subgraph = 
    get_subgraph(p.raw_graph, transform(n, [](PatternNode const &pn) { return pn.raw_node; })).graph;
  return UnlabelledGraphPattern{
    raw_subgraph,
  };
}



} // namespace FlexFlow
