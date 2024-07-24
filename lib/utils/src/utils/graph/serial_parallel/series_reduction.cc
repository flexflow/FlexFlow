#include "utils/graph/serial_parallel/series_reduction.h"
#include "utils/graph/multidigraph/algorithms/get_edges.h"
#include "utils/graph/multidigraph/algorithms/get_incoming_edges.h"
#include "utils/graph/multidigraph/algorithms/get_outgoing_edges.h"

namespace FlexFlow {

SeriesReduction make_series_reduction(MultiDiEdge const &e1, MultiDiEdge const &e2) {
  return SeriesReduction{e1, e2};
}

std::optional<SeriesReduction> find_series_reduction(MultiDiGraphView const &g) {
  std::unordered_set<MultiDiEdge> edges = get_edges(g);

  for (MultiDiEdge const &e1 : edges) {
    for (MultiDiEdge const &e2 : edges) {
      if (e1 == e2) {
        continue;
      }
      Node e1_dst = g.get_multidiedge_dst(e1);
      Node e2_src = g.get_multidiedge_src(e2);
      if (e1_dst != e2_src) {
        continue;
      }

      std::unordered_set<MultiDiEdge> outgoing = get_outgoing_edges(g, e1_dst);
      std::unordered_set<MultiDiEdge> incoming = get_incoming_edges(g, e1_dst);

      if (outgoing.size() > 1 || incoming.size() > 1) {
        continue;
      }

      return SeriesReduction{e1, e2};
    }
  }

  return std::nullopt;
}

} // namespace FlexFlow
