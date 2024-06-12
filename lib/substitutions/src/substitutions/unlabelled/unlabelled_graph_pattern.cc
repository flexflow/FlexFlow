#include "substitutions/unlabelled/unlabelled_graph_pattern.h"
#include "utils/containers.h"

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

std::unordered_set<PatternEdge> get_edges(UnlabelledGraphPattern const &p) {
  return transform(get_edges(p.raw_graph),
                   [](OpenMultiDiEdge const &e) { return PatternEdge{e}; });
}

std::vector<PatternNode>
    get_topological_ordering(UnlabelledGraphPattern const &p) {
  return transform(get_topological_ordering(p.raw_graph),
                   [](Node const &n) { return PatternNode{n}; });
}

UnlabelledGraphPattern get_subgraph(UnlabelledGraphPattern const &p,
                                    std::unordered_set<PatternNode> const &n) {
  MultiDiGraphView view = get_subgraph(
      p.raw_graph,
      transform(n, [](PatternNode const &n) { return n.raw_node; }));
  OpenMultiDiGraphView open_view = as_openmultidigraph(view);
  return UnlabelledGraphPattern{open_view};
}

std::unordered_set<UpwardOpenPatternEdge>
    get_incoming_edges(UnlabelledGraphPattern const &p, PatternNode const &n) {
  return transform(
      get_incoming_edges(p.raw_graph, n.raw_node),
      [](UpwardOpenMultiDiEdge const &e) { return UpwardOpenPatternEdge{e}; });
}

std::unordered_set<DownwardOpenPatternEdge>
    get_outgoing_edges(UnlabelledGraphPattern const &p, PatternNode const &n) {
  return transform(get_outgoing_edges(p.raw_graph, n.raw_node),
                   [](DownwardOpenMultiDiEdge const &e) {
                     return DownwardOpenPatternEdge{e};
                   });
}

} // namespace FlexFlow
