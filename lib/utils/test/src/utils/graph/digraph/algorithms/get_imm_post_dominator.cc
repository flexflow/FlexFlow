#include <doctest/doctest.h>
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_imm_post_dominator.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/containers/generate_map.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_imm_post_dominator") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();

    SUBCASE("trivial sequential graph") {
      std::vector<Node> n = add_nodes(g, 2);

      g.add_edge(DirectedEdge{n.at(0), n.at(1)});

      SUBCASE("n.at(0)") {
        std::optional<Node> result = get_imm_post_dominator(g, n.at(0));
        std::optional<Node> correct = n.at(1);
        CHECK(result == correct);
      }

      SUBCASE("n.at(1)") {
        std::optional<Node> result = get_imm_post_dominator(g, n.at(1));
        std::optional<Node> correct = std::nullopt;
        CHECK(result == correct);
      }
    }

    SUBCASE("actual non-straight-line graph") {
      std::vector<Node> n = add_nodes(g, 9);

      add_edges(g, {
        DirectedEdge{n.at(0), n.at(1)},
        DirectedEdge{n.at(0), n.at(2)},
        DirectedEdge{n.at(1), n.at(3)},
        DirectedEdge{n.at(1), n.at(4)},
        DirectedEdge{n.at(2), n.at(4)},
        DirectedEdge{n.at(2), n.at(5)},
        DirectedEdge{n.at(3), n.at(6)},
        DirectedEdge{n.at(4), n.at(6)},
        DirectedEdge{n.at(4), n.at(7)},
        DirectedEdge{n.at(5), n.at(7)},
        DirectedEdge{n.at(6), n.at(8)},
        DirectedEdge{n.at(7), n.at(8)},
      });

      std::unordered_map<Node, std::optional<Node>> result = generate_map(n, [&](Node const &nn) { return get_imm_post_dominator(g, nn); });
      std::unordered_map<Node, std::optional<Node>> correct = {
        {n.at(0), n.at(8)},
        {n.at(1), n.at(6)},
        {n.at(2), n.at(7)},
        {n.at(3), n.at(6)},
        {n.at(4), n.at(8)},
        {n.at(5), n.at(7)},
        {n.at(6), n.at(8)},
        {n.at(7), n.at(8)},
        {n.at(8), std::nullopt},
      };

      CHECK(result == correct);
    }

    // TODO add a cyclic graph
  }
}
