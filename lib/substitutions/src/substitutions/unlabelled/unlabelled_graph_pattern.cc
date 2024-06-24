#include "substitutions/unlabelled/unlabelled_graph_pattern.h"
#include "utils/containers.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/open_dataflow_graph/algorithms.h"
#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/digraph/algorithms.h"

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
                   [](OpenDataflowValue const &v) { return PatternValue{v}; });
}

std::vector<PatternNode> get_topological_ordering(UnlabelledGraphPattern const &p) {
  return transform(get_topological_ordering(p.raw_graph),
                   [](Node const &n) { return PatternNode{n}; });
}

std::vector<PatternValue>
    get_inputs_to_pattern_node(UnlabelledGraphPattern const &p, PatternNode const &n) {
  return transform(get_inputs(p.raw_graph, n.raw_node),
                   [](OpenDataflowValue const &v) { return PatternValue{v}; });
}

std::vector<PatternValue>
    get_outputs_from_pattern_node(UnlabelledGraphPattern const &p, PatternNode const &n) {
  return transform(get_outputs(p.raw_graph, n.raw_node),
                   [](DataflowOutput const &o) { return PatternValue{OpenDataflowValue{o}}; });
}

UnlabelledGraphPattern get_subgraph(UnlabelledGraphPattern const &p,
                                    std::unordered_set<PatternNode> const &n) {
  NOT_IMPLEMENTED();
  // return UnlabelledGraphPattern{
  //   get_subgraph(p.raw_graph,
  //                transform(n, [](PatternNode const &n) { return n.raw_node; }));
  // };
}


} // namespace FlexFlow
