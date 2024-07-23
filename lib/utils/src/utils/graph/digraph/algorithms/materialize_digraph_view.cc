#include "utils/graph/digraph/algorithms/materialize_digraph_view.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/digraph/algorithms.h"

namespace FlexFlow {

void materialize_digraph_view(DiGraph &result, DiGraphView const &g) {
  for (Node const &n : get_nodes(g)) {
    result.add_node_unsafe(n);
  }
  for (DirectedEdge const &e : get_edges(g)) {
    result.add_edge(e); 
  }
}

} // namespace FlexFlow
