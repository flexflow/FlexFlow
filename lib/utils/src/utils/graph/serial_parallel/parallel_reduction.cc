#include "utils/graph/serial_parallel/parallel_reduction.h"
#include "utils/graph/multidigraph/algorithms/get_edges.h"

namespace FlexFlow {

ParallelReduction make_parallel_reduction(MultiDiEdge const &e1, MultiDiEdge const &e2) {
  return ParallelReduction{{e1, e2}}; 
}

std::optional<ParallelReduction> find_parallel_reduction(MultiDiGraphView const &g) {
  std::unordered_set<MultiDiEdge> edges = get_edges(g);

  for (MultiDiEdge const &e1 : edges) {
    for (MultiDiEdge const &e2 : edges) {
      if (e1 != e2
          && g.get_multidiedge_src(e1) == g.get_multidiedge_src(e2) 
          && g.get_multidiedge_dst(e1) == g.get_multidiedge_dst(e2)) {
        return make_parallel_reduction(e1, e2);
      }
    }
  }

  return std::nullopt;
}

MultiDiEdge apply_parallel_reduction(MultiDiGraph &g, ParallelReduction const &r) {
  g.remove_edge(r.edges.max());
  return r.edges.min();
}

} // namespace FlexFlow
