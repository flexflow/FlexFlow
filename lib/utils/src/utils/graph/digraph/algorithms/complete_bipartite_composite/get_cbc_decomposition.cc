#include "utils/graph/digraph/algorithms/complete_bipartite_composite/get_cbc_decomposition.h"
#include "utils/containers/are_disjoint.h"
#include "utils/containers/extend.h"
#include "utils/containers/get_first.h"
#include "utils/containers/set_minus.h"
#include "utils/containers/set_of.h"
#include "utils/containers/values.h"
#include "utils/containers/vector_of.h"
#include "utils/fmt/set.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/digraph/algorithms/complete_bipartite_composite/is_complete_bipartite_digraph.h"
#include "utils/graph/digraph/algorithms/get_incoming_edges.h"
#include "utils/graph/digraph/algorithms/get_outgoing_edges.h"
#include "utils/graph/digraph/algorithms/get_predecessors.h"
#include "utils/graph/digraph/algorithms/get_successors.h"
#include "utils/graph/digraph/algorithms/get_weakly_connected_components.h"
#include "utils/graph/node/algorithms.h"
#include "utils/hash/unordered_set.h"
#include <queue>

namespace FlexFlow {

std::optional<CompleteBipartiteCompositeDecomposition>
    get_cbc_decomposition_with_edge_order_internal(
        DiGraphView const &g, std::vector<DirectedEdge> const &edge_order) {
  // implementation of the algorithm from https://doi.org/10.1145/800135.804393
  // top left of page 8, second paragraph

  std::queue<DirectedEdge> edges_to_process;
  for (DirectedEdge const &e : edge_order) {
    edges_to_process.push(e);
  }

  std::unordered_set<Node> already_in_a_head = {};
  std::unordered_set<Node> already_in_a_tail = {};

  std::unordered_set<DirectedEdge> already_processed = {};

  CompleteBipartiteCompositeDecomposition result =
      CompleteBipartiteCompositeDecomposition{{}};

  while (!edges_to_process.empty()) {
    DirectedEdge e = edges_to_process.front();
    edges_to_process.pop();
    if (contains(already_processed, e)) {
      continue;
    }

    std::unordered_set<Node> head = get_predecessors(g, e.dst);
    std::unordered_set<Node> tail = get_successors(g, e.src);

    if (!are_disjoint(head, tail)) {
      return std::nullopt;
    }

    std::unordered_set<DirectedEdge> from_head_to_tail =
        g.query_edges(DirectedEdgeQuery{head, tail});

    DiGraphView subgraph = get_subgraph(g, set_union(head, tail));
    if (!is_complete_bipartite_digraph(subgraph, head)) {
      return std::nullopt;
    }

    if (set_union(values(get_outgoing_edges(g, head))) != from_head_to_tail) {
      return std::nullopt;
    }
    if (set_union(values(get_incoming_edges(g, tail))) != from_head_to_tail) {
      return std::nullopt;
    }

    result.subgraphs.insert(BipartiteComponent{head, tail});
    already_processed = set_union(already_processed, from_head_to_tail);
    extend(already_in_a_head, head);
    extend(already_in_a_tail, tail);
  }

  assert(already_in_a_head == set_minus(get_nodes(g), get_sinks(g)));
  assert(already_in_a_tail == set_minus(get_nodes(g), get_sources(g)));

  return result;
}

std::optional<CompleteBipartiteCompositeDecomposition>
    get_cbc_decomposition(DiGraphView const &g) {
  std::vector<DirectedEdge> edge_order = vector_of(get_edges(g));
  return get_cbc_decomposition_with_edge_order_internal(g, edge_order);
}

} // namespace FlexFlow
