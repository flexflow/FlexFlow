#include "utils/graph/digraph/algorithms/inverse_line_graph/get_inverse_line_graph.h"
#include "utils/bidict/generate_bidict.h"
#include "utils/containers/set_minus.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/digraph/algorithms/complete_bipartite_composite/complete_bipartite_composite_decomposition.h"
#include "utils/graph/digraph/algorithms/complete_bipartite_composite/get_cbc_decomposition.h"
#include "utils/graph/digraph/digraph.h"
#include "utils/graph/instances/adjacency_multidigraph.h"
#include "utils/graph/node/algorithms.h"
#include "utils/hash/pair.h"

namespace FlexFlow {

std::optional<InverseLineGraphResult> get_inverse_line_graph(DiGraphView const &view) {
  // implementation of the algorithm from https://doi.org/10.1145/800135.804393
  // left of page 8, definition 5
  MultiDiGraph result_graph = MultiDiGraph::create<AdjacencyMultiDiGraph>();

  CompleteBipartiteCompositeDecomposition cbc_decomposition = ({
    std::optional<CompleteBipartiteCompositeDecomposition> maybe_decomp = get_cbc_decomposition(view);
    if (!maybe_decomp.has_value()) {
      return std::nullopt;
    }

    maybe_decomp.value();
  });

  Node alpha = result_graph.add_node();
  Node omega = result_graph.add_node();

  bidict<BipartiteComponent, Node> component_nodes = generate_bidict(
      cbc_decomposition.subgraphs,
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

  auto src_for_node = [&](Node const &v) -> Node {
    if (contains(sources, v)) {
      return alpha;
    } else {
      return component_nodes.at_l(t(v));
    }
  };

  auto dst_for_node = [&](Node const &v) -> Node {
    if (contains(sinks, v)) {
      return omega;
    } else {
      return component_nodes.at_l(h(v));
    }
  };

  bidict<MultiDiEdge, Node> inverse_edge_to_line_node_bidict;

  for (Node const &v : get_nodes(view)) {
    MultiDiEdge e = result_graph.add_edge(src_for_node(v), dst_for_node(v));

    assert(!inverse_edge_to_line_node_bidict.contains_r(v));
    inverse_edge_to_line_node_bidict.equate({e, v});
  }

  return InverseLineGraphResult{
      result_graph,
      inverse_edge_to_line_node_bidict,
  };
}

} // namespace FlexFlow
