#include "utils/graph/digraph/algorithms/inverse_line_graph/get_inverse_line_graph.h"
#include "utils/bidict/generate_bidict.h"
#include "utils/containers/set_minus.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/digraph/algorithms/complete_bipartite_composite/complete_bipartite_composite_decomposition.h"
#include "utils/graph/digraph/algorithms/complete_bipartite_composite/get_cbc_decomposition.h"
#include "utils/graph/digraph/digraph.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/node/algorithms.h"
#include "utils/hash/pair.h"

namespace FlexFlow {

InverseLineGraphResult get_inverse_line_graph(DiGraphView const &view) {
  // implementation of the algorithm from https://doi.org/10.1145/800135.804393 left of page 8, definition 5
  DiGraph result_graph = DiGraph::create<AdjacencyDiGraph>();

  CompleteBipartiteCompositeDecomposition cbc_decomposition = unwrap(get_cbc_decomposition(view), [] {
    throw mk_runtime_error("get_inverse_line_graph requires a cbc graph");
  });

  Node alpha = result_graph.add_node();
  Node omega = result_graph.add_node();
  // std::unordered_set<std::unordered_set<Node>> all_subcomponents = set_union(get_head_subcomponents(cbc_decomposition), get_tail_subcomponents(cbc_decomposition));
  // bidict<std::unordered_set<Node>, Node> subcomponent_nodes = generate_bidict(all_subcomponents, 
  //                                                                             [&](std::unordered_set<Node> const &) { return result_graph.add_node(); });
  bidict<BipartiteComponent, Node> component_nodes = generate_bidict(cbc_decomposition.subgraphs,
                                                                     [&](BipartiteComponent const &) { return result_graph.add_node(); });

  // h and t notation to match paper
  auto h = [&](Node const &n) -> BipartiteComponent {
    return get_component_containing_node_in_head(cbc_decomposition, n).value();
  };
  auto t = [&](Node const &n) -> BipartiteComponent {
    return get_component_containing_node_in_tail(cbc_decomposition, n).value();
  };

  std::unordered_set<Node> sources = get_sources(view);
  std::unordered_set<Node> sinks = get_sinks(view);

  auto edge_for_node = [&](Node const &v) -> DirectedEdge {
    if (contains(sources, v) && contains(sinks, v)) {
      return DirectedEdge{alpha, omega};
    } else if (contains(sources, v)) {
      return DirectedEdge{alpha, component_nodes.at_l(h(v))};
    } else if (contains(sinks, v)) {
      return DirectedEdge{component_nodes.at_l(t(v)), omega};
    } else {
      return DirectedEdge{component_nodes.at_l(t(v)), component_nodes.at_l(h(v))};
    }
  };

  bidict<DirectedEdge, Node> inverse_edge_to_line_node_bidict;

  for (Node const &v : get_nodes(view)) {
    DirectedEdge e = edge_for_node(v);
    result_graph.add_edge(e);

    assert (!inverse_edge_to_line_node_bidict.contains_l(e));
    assert (!inverse_edge_to_line_node_bidict.contains_r(v));
    inverse_edge_to_line_node_bidict.equate({e, v});
  }

  return InverseLineGraphResult{
    result_graph,
    inverse_edge_to_line_node_bidict,
  };
}

} // namespace FlexFlow
