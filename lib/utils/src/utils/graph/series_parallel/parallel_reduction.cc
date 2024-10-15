#include "utils/graph/series_parallel/parallel_reduction.h"
#include "utils/graph/multidigraph/algorithms/get_edge_counts.h"
#include "utils/graph/multidigraph/algorithms/get_edges.h"
#include "utils/graph/multidigraph/algorithms/get_incoming_edges.h"
#include "utils/graph/multidigraph/algorithms/get_outgoing_edges.h"
#include "utils/graph/node/algorithms.h"

namespace FlexFlow {

ParallelReduction make_parallel_reduction(MultiDiEdge const &e1,
                                          MultiDiEdge const &e2) {
  return ParallelReduction{{e1, e2}};
}

std::optional<ParallelReduction>
    find_parallel_reduction(MultiDiGraphView const &g) {

  for (auto const &[directed_edge, count] : get_edge_counts(g)) {

    if (count <= 1) {
      continue;
    }

    std::unordered_set<MultiDiEdge> const &outgoing_edges =
        get_outgoing_edges(g, directed_edge.src);
    for (MultiDiEdge const &e1 : outgoing_edges) {
      for (MultiDiEdge const &e2 : outgoing_edges) {
        if (e1 != e2 &&
            g.get_multidiedge_dst(e1) == g.get_multidiedge_dst(e2)) {
          return make_parallel_reduction(e1, e2);
        }
      }
    }
  }

  return std::nullopt;
}

MultiDiEdge apply_parallel_reduction(MultiDiGraph &g,
                                     ParallelReduction const &r) {
  g.remove_edge(r.edges.max());
  return r.edges.min();
}

} // namespace FlexFlow
