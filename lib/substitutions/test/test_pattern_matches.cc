#include "doctest.h"
#include "graph_pattern_match.h"
#include "rapidcheck.h"

using namespace FlexFlow;

struct AlwaysTrue {
  template <typename T>
  bool operator()(T const &t) const {
    return true;
  }
};

MultiDiGraph
    construct_multidigraph(int num_nodes,
                           std::vector<std::pair<int, int>> const &edges) {
  MultiDiGraph g = MultiDiGraph::create<AdjacencyMultiDiGraph>();

  std::vector<Node> nodes;
  for (int i = 0; i < num_nodes; ++i) {
    nodes.push_back(g.add_node());
  }

  for (std::pair<int, int> e : edges) {
    if (e.first > e.second) {
      std::swap(e.first, e.second);
    }

    g.add_edge(MultiDiEdge{
        nodes[e.first], nodes[e.second], g.add_node_port(), g.add_node_port()});
  }

  return g;
}

namespace rc {

template <>
struct Arbitrary<MultiDiGraph> {
  static int const MAX_GRAPH_SIZE = 200;

  static Gen<MultiDiGraph> arbitrary() {
    auto gen_edges = [](int num_nodes) {
      return gen::container<std::vector<std::pair<int, int>>>(
          gen::inRange(0, num_nodes));
    };

    auto gen_graph = [&](int num_nodes) {
      return gen::apply(
          [=](std::vector<std::pair<int, int>> const &edges) {
            return construct_multidigraph(num_nodes, edges);
          },
          gen_edges(num_nodes));
    };

    return gen::apply(gen_graph, gen::inRange(1, MAX_GRAPH_SIZE));
  }
};

} // namespace rc

TEST_CASE("find_pattern_matches") {
  rc::check([](MultiDiGraph const &g) {
    std::unordered_set<Node> subgraph_nodes =
        *rc::gen::container<std::unordered_set<Node>>(
            rc::gen::elementOf(get_nodes(g)));
    OpenMultiDiGraphView subgraph =
        get_subgraph(as_openmultidigraph(g), subgraph_nodes);

    std::unordered_set<MultiDiGraphPatternMatch> matches =
        find_pattern_matches(subgraph, g, AlwaysTrue{});

    RC_ASSERT(!matches.empty());

    for (MultiDiGraphPatternMatch const &match : matches) {
      RC_ASSERT(pattern_matches(subgraph, g, match, AlwaysTrue{}));
    }
  });
}
