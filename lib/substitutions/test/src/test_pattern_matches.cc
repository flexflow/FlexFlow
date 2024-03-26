#include "doctest/doctest.h"
#include "rapidcheck.h"
#include "substitutions/graph_pattern_match.h"
#include "test/utils/all.h"

using namespace FlexFlow;

namespace rc {

template <>
struct Arbitrary<MultiDiGraph> {
  static int const MAX_GRAPH_SIZE = 200;
  static int const MAX_EDGE_SIZE = 1000;

  static Gen<MultiDiGraph> arbitrary() {
    return gen::exec([&] {
      int num_nodes = *gen::inRange(1, MAX_GRAPH_SIZE + 1);
      MultiDiGraph g = MultiDiGraph::template create<AdjacencyMultiDiGraph>();

      std::vector<Node> nodes;
      for (int i = 0; i < num_nodes; ++i) {
        nodes.push_back(g.add_node());
      }

      int num_edges = *gen::inRange(1, MAX_GRAPH_SIZE + 1);
      for (int i = 0; i < num_edges; ++i) {
        int src_id = *gen::inRange(0, num_nodes);
        int dst_id = *gen::inRange(0, num_nodes);
        if (src_id > dst_id) {
          std::swap(src_id, dst_id);
        }

        g.add_edge(MultiDiEdge{nodes[dst_id],
                               g.add_node_port(),
                               nodes[src_id],
                               g.add_node_port()});
      }

      return g;
    });
  }
};

} // namespace rc

// TEST_CASE("find_pattern_matches") {
//   rc::check([](MultiDiGraph const &g) {
//     std::unordered_set<Node> subgraph_nodes = *rc::subset_of(get_nodes(g));
//     OpenMultiDiGraphView subgraph =
//         get_subgraph<OpenMultiDiSubgraphView>(as_openmultidigraph(g),
//         subgraph_nodes);

//     std::vector<MultiDiGraphPatternMatch> matches =
//         find_pattern_matches(subgraph, as_openmultidigraph(g), AlwaysTrue{});

//     RC_ASSERT(!matches.empty());

//     for (MultiDiGraphPatternMatch const &match : matches) {
//       RC_ASSERT(pattern_matches(subgraph, as_openmultidigraph(g), match,
//       AlwaysTrue{}));
//     }
//   });
// }

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("find_pattern_matches_small") {
    MultiDiGraph g = MultiDiGraph::template create<AdjacencyMultiDiGraph>();

    {
      Node n0 = g.add_node();
      Node n1 = g.add_node();
      Node n2 = g.add_node();
      Node n3 = g.add_node();

      MultiDiEdge e0{n1, g.add_node_port(), n0, g.add_node_port()};
      MultiDiEdge e1{n2, g.add_node_port(), n1, g.add_node_port()};
      MultiDiEdge e2{n3, g.add_node_port(), n2, g.add_node_port()};

      g.add_edge(e0);
      g.add_edge(e1);
      g.add_edge(e2);
    }

    MultiDiGraph sg0 = MultiDiGraph::template create<AdjacencyMultiDiGraph>();

    {
      Node n0 = sg0.add_node();
      Node n1 = sg0.add_node();

      MultiDiEdge e0{n1, sg0.add_node_port(), n0, sg0.add_node_port()};

      sg0.add_edge(e0);
    }

    MatchAdditionalCriterion always_true{
        [](Node const &, Node const &) { return true; },
        [](OpenMultiDiEdge const &, OpenMultiDiEdge const &) { return true; }};

    std::vector<MultiDiGraphPatternMatch> matches = find_pattern_matches(
        as_openmultidigraph(sg0), as_openmultidigraph(g), always_true);

    RC_ASSERT(matches.size() == 3);

    for (MultiDiGraphPatternMatch const &match : matches) {
      RC_ASSERT(pattern_matches(
          as_openmultidigraph(sg0), as_openmultidigraph(g), match, always_true));
    }
  }
}
