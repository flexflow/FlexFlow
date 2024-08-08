#include "doctest/doctest.h"
#include "rapidcheck.h"
#include "substitutions/unlabelled/find_pattern_matches.h"
#include "substitutions/unlabelled/match_additional_criterion.h"
#include "substitutions/unlabelled/pattern_matching.h"
#include "test/utils/all.h"
#include "utils/containers/get_only.h"
#include "utils/graph/instances/unordered_set_dataflow_graph.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/open_dataflow_graph/algorithms.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_subgraph.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_subgraph_inputs.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_graph.h"
#include "utils/overload.h"

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
//       MultiDiGraph g = MultiDiGraph::template
//       create<AdjacencyMultiDiGraph>();
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
    OpenDataflowGraph pattern_graph =
        OpenDataflowGraph::create<UnorderedSetDataflowGraph>();

    NodeAddedResult pattern_n0_added = pattern_graph.add_node({}, 1);
    Node pattern_n0 = pattern_n0_added.node;
    OpenDataflowValue pattern_v0 =
        OpenDataflowValue{get_only(pattern_n0_added.outputs)};

    NodeAddedResult pattern_n1_added = pattern_graph.add_node({pattern_v0}, 1);
    Node pattern_n1 = pattern_n1_added.node;
    OpenDataflowValue pattern_v1 =
        OpenDataflowValue{get_only(pattern_n1_added.outputs)};

    UnlabelledGraphPattern pattern = UnlabelledGraphPattern{pattern_graph};
    PatternNode p0 = PatternNode{pattern_n0};
    PatternNode p1 = PatternNode{pattern_n1};

    OpenDataflowGraph graph =
        OpenDataflowGraph::create<UnorderedSetDataflowGraph>();

    NodeAddedResult n0_added = graph.add_node({}, 1);
    Node n0 = n0_added.node;
    OpenDataflowValue v0 = OpenDataflowValue{get_only(n0_added.outputs)};

    NodeAddedResult n1_added = graph.add_node({v0}, 1);
    Node n1 = n1_added.node;
    OpenDataflowValue v1 = OpenDataflowValue{get_only(n1_added.outputs)};

    NodeAddedResult n2_added = graph.add_node({v1}, 1);
    Node n2 = n2_added.node;
    OpenDataflowValue v2 = OpenDataflowValue{get_only(n2_added.outputs)};

    NodeAddedResult n3_added = graph.add_node({v2}, 1);
    Node n3 = n3_added.node;
    OpenDataflowValue v3 = OpenDataflowValue{get_only(n3_added.outputs)};

    UnlabelledDataflowGraphPatternMatch match =
        UnlabelledDataflowGraphPatternMatch{
            bidict<PatternNode, Node>{
                {p0, n0},
                {p1, n1},
            },
            bidict<PatternInput, OpenDataflowValue>{}};

    UnlabelledDataflowGraphPatternMatch invalid_match =
        UnlabelledDataflowGraphPatternMatch{
            bidict<PatternNode, Node>{
                {p0, n1},
                {p1, n2},
            },
            bidict<PatternInput, OpenDataflowValue>{}};

    std::vector<OpenDataflowEdge> n1_incoming = {OpenDataflowEdge{
        DataflowEdge{
            DataflowOutput{n0, 0},
            DataflowInput{n1, 0},
        },
    }};

    SUBCASE("get_incoming_edges") {
      SUBCASE("n0") {
        std::vector<OpenDataflowEdge> result = get_incoming_edges(graph, n0);
        std::vector<OpenDataflowEdge> correct = {};
        CHECK(result == correct);
      }
      SUBCASE("n1") {
        std::vector<OpenDataflowEdge> result = get_incoming_edges(graph, n1);
        std::vector<OpenDataflowEdge> correct = n1_incoming;
        CHECK(result == correct);
      }
      SUBCASE("both") {
        std::unordered_map<Node, std::vector<OpenDataflowEdge>> result =
            get_incoming_edges(graph, {n0, n1});
        std::unordered_map<Node, std::vector<OpenDataflowEdge>> correct = {
            {n0, {}}, {n1, n1_incoming}};
        CHECK(result == correct);
      }
    }

    SUBCASE("get_subgraph_inputs") {
      std::unordered_set<OpenDataflowValue> result =
          get_subgraph_inputs(graph, {n0, n1});
      std::unordered_set<OpenDataflowValue> correct = {};
      CHECK(result == correct);
    }

    SUBCASE("get_subgraph") {
      OpenDataflowGraphView g = get_subgraph(graph, {n0, n1}).graph;
      SUBCASE("nodes") {
        std::unordered_set<Node> result = get_nodes(g);
        std::unordered_set<Node> correct = {n0, n1};
        CHECK(result == correct);
      }
      SUBCASE("inputs") {
        std::unordered_set<DataflowGraphInput> result = g.get_inputs();
        std::unordered_set<DataflowGraphInput> correct = {};
        CHECK(result == correct);
      }
      SUBCASE("get_open_dataflow_values") {
        std::unordered_set<OpenDataflowValue> values =
            get_open_dataflow_values(g);
        CHECK(values.size() == 2);
      }
    }

    SUBCASE("subgraph_matched") {
      OpenDataflowGraphView result = subgraph_matched(graph, match).graph;
      std::unordered_set<Node> result_nodes = get_nodes(result);
      std::unordered_set<Node> correct_nodes = {n0, n1};
      CHECK(result_nodes == correct_nodes);
    }

    SUBCASE("unlabelled_pattern_does_match") {
      CHECK(unlabelled_pattern_does_match(
          pattern, graph, match, match_additional_crition_always_true()));
      CHECK_FALSE(unlabelled_pattern_does_match(
          pattern,
          graph,
          invalid_match,
          match_additional_crition_always_true()));
    }

    SUBCASE("unlabelled_pattern_does_match (open)") {
      OpenDataflowGraph g =
          OpenDataflowGraph::create<UnorderedSetDataflowGraph>();
      DataflowGraphInput i0 = g.add_input();

      NodeAddedResult g_n0_added = g.add_node({OpenDataflowValue{i0}}, 1);
      Node g_n0 = g_n0_added.node;
      OpenDataflowValue g_v0 = OpenDataflowValue{get_only(g_n0_added.outputs)};
      PatternNode g_p0 = PatternNode{g_n0};
      PatternInput g_pi0 = PatternInput{i0};

      UnlabelledGraphPattern open_pattern = UnlabelledGraphPattern{g};

      UnlabelledDataflowGraphPatternMatch open_match =
          UnlabelledDataflowGraphPatternMatch{
              bidict<PatternNode, Node>{
                  {g_p0, n1},
              },
              bidict<PatternInput, OpenDataflowValue>{
                  {g_pi0, v0},
              }};
      CHECK(unlabelled_pattern_does_match(
          open_pattern,
          graph,
          open_match,
          match_additional_crition_always_true()));
    }

    SUBCASE("find_pattern_matches") {
      std::vector<UnlabelledDataflowGraphPatternMatch> matches =
          find_pattern_matches(
              pattern, graph, match_additional_crition_always_true());
      std::vector<UnlabelledDataflowGraphPatternMatch> correct = {match};

      CHECK(matches == correct);
    }
  }
}
