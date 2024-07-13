#include "doctest/doctest.h"
#include "rapidcheck.h"
#include "substitutions/unlabelled/find_pattern_matches.h"
#include "test/utils/all.h"
#include "substitutions/unlabelled/match_additional_criterion.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_graph.h"
#include "utils/graph/instances/unordered_set_dataflow_graph.h"
#include "substitutions/unlabelled/pattern_matching.h"

using namespace FlexFlow;

namespace rc {

// template <>
// struct Arbitrary<MultiDiGraph> {
//   static int const MAX_GRAPH_SIZE = 200;
//   static int const MAX_EDGE_SIZE = 1000;
//
//   static Gen<MultiDiGraph> arbitrary() {
//     return gen::exec([&] {
//       int num_nodes = *gen::inRange(1, MAX_GRAPH_SIZE + 1);
//       MultiDiGraph g = MultiDiGraph::template create<AdjacencyMultiDiGraph>();
//
//       std::vector<Node> nodes;
//       for (int i = 0; i < num_nodes; ++i) {
//         nodes.push_back(g.add_node());
//       }
//
//       int num_edges = *gen::inRange(1, MAX_GRAPH_SIZE + 1);
//       for (int i = 0; i < num_edges; ++i) {
//         int src_id = *gen::inRange(0, num_nodes);
//         int dst_id = *gen::inRange(0, num_nodes);
//         if (src_id > dst_id) {
//           std::swap(src_id, dst_id);
//         }
//
//         g.add_edge(MultiDiEdge{nodes[dst_id],
//                                g.add_node_port(),
//                                nodes[src_id],
//                                g.add_node_port()});
//       }
//
//       return g;
//     });
//   }
// };

} // namespace rc

// TEST_CASE("find_pattern_matches") {
//   RC_SUBCASE([](MultiDiGraph const &g) {
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
    UnlabelledGraphPattern pattern = [] {
      OpenDataflowGraph g = OpenDataflowGraph::create<UnorderedSetDataflowGraph>();

      NodeAddedResult n0_added = g.add_node({}, 1);
      Node n0 = n0_added.node;
      OpenDataflowValue v0 = OpenDataflowValue{get_only(n0_added.outputs)};

      NodeAddedResult n1_added = g.add_node({v0}, 1);
      Node n1 = n1_added.node;
      OpenDataflowValue v1 = OpenDataflowValue{get_only(n1_added.outputs)};

      return UnlabelledGraphPattern{g};
    }();

    OpenDataflowGraph graph = [] {
      OpenDataflowGraph g = OpenDataflowGraph::create<UnorderedSetDataflowGraph>();
      
      NodeAddedResult n0_added = g.add_node({}, 1);
      Node n0 = n0_added.node;
      OpenDataflowValue v0 = OpenDataflowValue{get_only(n0_added.outputs)};

      NodeAddedResult n1_added = g.add_node({v0}, 1);
      Node n1 = n1_added.node;
      OpenDataflowValue v1 = OpenDataflowValue{get_only(n1_added.outputs)};

      NodeAddedResult n2_added = g.add_node({v1}, 1);
      Node n2 = n2_added.node;
      OpenDataflowValue v2 = OpenDataflowValue{get_only(n2_added.outputs)};

      NodeAddedResult n3_added = g.add_node({v2}, 1);
      Node n3 = n3_added.node;
      OpenDataflowValue v3 = OpenDataflowValue{get_only(n3_added.outputs)};

      return g;
    }();

    std::vector<UnlabelledDataflowGraphPatternMatch> matches = find_pattern_matches(
        pattern, graph, match_additional_crition_always_true());

    CHECK(matches.size() == 3);

    for (UnlabelledDataflowGraphPatternMatch const &match : matches) {
      CHECK(unlabelled_pattern_does_match(pattern,
                                          graph,
                                          match,
                                          match_additional_crition_always_true()));
    }
  }
}
