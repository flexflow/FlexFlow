#include "utils/graph/digraph/algorithms/complete_bipartite_composite/get_cbc_decomposition.h"
#include "utils/containers/are_disjoint.h"
#include "utils/containers/extend.h"
#include "utils/containers/get_first.h"
#include "utils/containers/set_minus.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_incoming_edges.h"
#include "utils/graph/digraph/algorithms/get_outgoing_edges.h"
#include "utils/graph/digraph/algorithms/get_predecessors.h"
#include "utils/graph/digraph/algorithms/get_successors.h"
#include "utils/containers/values.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/digraph/algorithms/get_weakly_connected_components.h"
#include "utils/hash/unordered_set.h"

namespace FlexFlow {

std::optional<CompleteBipartiteCompositeDecomposition> get_cbc_decomposition(DiGraphView const &g) {
  // implementation of the algorithm from https://doi.org/10.1145/800135.804393 top left of page 8, second paragraph

  assert(get_weakly_connected_components(g).size() == 1); // possible to handle other cases, but for now we don't

  std::unordered_set<Node> already_in_a_head = {};
  std::unordered_set<Node> already_in_a_tail = {};
  std::unordered_set<DirectedEdge> edges_to_process = get_edges(g);

  CompleteBipartiteCompositeDecomposition result = CompleteBipartiteCompositeDecomposition{{}};

  while (!edges_to_process.empty()) {
    DirectedEdge e = get_first(edges_to_process);

    std::unordered_set<Node> head = get_predecessors(g, e.dst);
    std::unordered_set<Node> tail = get_successors(g, e.src);

    if (!are_disjoint(head, tail)) {
      return std::nullopt;
    }

    std::unordered_set<DirectedEdge> from_head_to_tail = g.query_edges(DirectedEdgeQuery{head, tail});
    if (set_union(values(get_outgoing_edges(g, head))) != from_head_to_tail) {
      return std::nullopt;
    }
    if (set_union(values(get_incoming_edges(g, tail))) != from_head_to_tail) {
      return std::nullopt;
    }

    result.subgraphs.insert(BipartiteComponent{head, tail});
    edges_to_process = set_minus(edges_to_process, from_head_to_tail);
    extend(already_in_a_head, head);
    extend(already_in_a_tail, tail);
  }

  assert (already_in_a_head == set_minus(get_nodes(g), get_sinks(g)));
  assert (already_in_a_tail == set_minus(get_nodes(g), get_sources(g)));

  return result;
}

} // namespace FlexFlow
