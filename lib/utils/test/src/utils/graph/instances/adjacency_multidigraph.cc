#include <doctest/doctest.h>
#include "utils/graph/instances/adjacency_multidigraph.h"
#include "utils/graph/multidigraph/multidiedge_query.h"
#include "utils/graph/node/node_query.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("AdjacencyMultiDiGraph") {
    MultiDiGraph g = MultiDiGraph::create<AdjacencyMultiDiGraph>();

    auto check_state = [&](std::unordered_set<Node> const &correct_nodes, std::unordered_set<MultiDiEdge> const &correct_edges) {
      {
        std::unordered_set<Node> result = g.query_nodes(node_query_all());
        std::unordered_set<Node> correct = correct_nodes;
        REQUIRE(result == correct);
      }

      {
        std::unordered_set<MultiDiEdge> result = g.query_edges(multidiedge_query_all());
        std::unordered_set<MultiDiEdge> correct = correct_edges;
        REQUIRE(result == correct);
      }
    };

    check_state({}, {});

    Node n1 = g.add_node();

    check_state({n1}, {});

    MultiDiEdge e1 = g.add_edge(n1, n1);

    check_state({n1}, {e1});
    {
      Node result = g.get_multidiedge_src(e1);
      Node correct = n1;
      REQUIRE(result == correct);
    }
    {
      Node result = g.get_multidiedge_dst(e1);
      Node correct = n1;
      REQUIRE(result == correct);
    }

    MultiDiEdge e2 = g.add_edge(n1, n1);

    check_state({n1}, {e1, e2});

    Node n2 = g.add_node();

    check_state({n1, n2}, {e1, e2});

    MultiDiEdge e3 = g.add_edge(n1, n2);

    check_state({n1, n2}, {e1, e2, e3});
    {
      Node result = g.get_multidiedge_src(e3);
      Node correct = n1;
      REQUIRE(result == correct);
    }
    {
      Node result = g.get_multidiedge_dst(e3);
      Node correct = n2;
      REQUIRE(result == correct);
    }

    MultiDiEdge e4 = g.add_edge(n2, n1);
    check_state({n1, n2}, {e1, e2, e3, e4});

    {
      MultiDiEdgeQuery input = MultiDiEdgeQuery{{n1}, query_set<Node>::matchall()};
      std::unordered_set<MultiDiEdge> result = g.query_edges(input);
      std::unordered_set<MultiDiEdge> correct = {e1, e2, e3};
      CHECK(result == correct);
    }

    {
      MultiDiEdgeQuery input = MultiDiEdgeQuery{query_set<Node>::matchall(), {n1}};
      std::unordered_set<MultiDiEdge> result = g.query_edges(input);
      std::unordered_set<MultiDiEdge> correct = {e1, e2, e4};
      CHECK(result == correct);
    }

    {
      MultiDiEdgeQuery input = MultiDiEdgeQuery{{n1}, {n2}};
      std::unordered_set<MultiDiEdge> result = g.query_edges(input);
      std::unordered_set<MultiDiEdge> correct = {e3};
      CHECK(result == correct);
    }

    {
      MultiDiEdgeQuery input = MultiDiEdgeQuery{{n1}, {n1}};
      std::unordered_set<MultiDiEdge> result = g.query_edges(input);
      std::unordered_set<MultiDiEdge> correct = {e1, e2};
      CHECK(result == correct);
    }

    SUBCASE("remove edge") {
      g.remove_edge(e3);
      check_state({n1, n2}, {e1, e2, e4});
    }

    SUBCASE("remove node") {
      g.remove_node(n2);
      check_state({n1}, {e1, e2});
      g.remove_node(n1);
      check_state({}, {});
    }

    SUBCASE("copy") {
      MultiDiGraphView g2 = g;
      SUBCASE("nodes") {
        g.add_node();
        std::unordered_set<Node> result = g2.query_nodes(node_query_all());
        std::unordered_set<Node> correct = {n1, n2};
        CHECK(result == correct);
      }
      SUBCASE("edges") {
        g.add_edge(n1, n2);
        std::unordered_set<MultiDiEdge> result = g2.query_edges(multidiedge_query_all());
        std::unordered_set<MultiDiEdge> correct = {e1, e2, e3, e4};
        CHECK(result == correct);
      }
    }

    // SUBCASE("create_copy_of") {
    //   MultiDiGraphView g2 = g;
    // }
  }
}
