#include "utils/graph/multidigraph/algorithms/get_edges.h"
#include "utils/graph/multidigraph/multidiedge_query.h"

namespace FlexFlow {

std::unordered_set<MultiDiEdge> get_edges(MultiDiGraphView const &g) {
  return g.query_edges(multidiedge_query_all());
}

} // namespace FlexFlow
