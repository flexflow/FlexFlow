#include "utils/graph/multidigraph/algorithms/get_edge_counts.h"
#include "utils/containers/as_vector.h"
#include "utils/containers/get_element_counts.h"
#include "utils/containers/transform.h"
#include "utils/graph/multidigraph/algorithms/get_directed_edge.h"
#include "utils/graph/multidigraph/algorithms/get_edges.h"

namespace FlexFlow {

std::unordered_map<DirectedEdge, int>
    get_edge_counts(MultiDiGraphView const &g) {
  return get_element_counts(
      transform(as_vector(get_edges(g)),
                [&](MultiDiEdge const &e) { return get_directed_edge(g, e); }));
}

} // namespace FlexFlow
