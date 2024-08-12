#include "utils/graph/multidigraph/multidigraph.h"
#include "test/utils/doctest.h"
#include "utils/containers/contains.h"
#include "utils/graph/instances/adjacency_multidigraph.h"
#include "utils/graph/multidigraph/multidiedge_query.h"
#include "utils/graph/query_set.h"
#include <unordered_set>
#include <vector>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("MultiDiGraph") {
    MultiDiGraph g = MultiDiGraph::create<AdjacencyMultiDiGraph>();

    Node n0 = g.add_node();
    Node n1 = g.add_node();
    Node n2 = g.add_node();

    MultiDiEdge e0 = g.add_edge(n1, n0);
    MultiDiEdge e1 = g.add_edge(n2, n0);
    MultiDiEdge e2 = g.add_edge(n0, n2);
    MultiDiEdge e3 = g.add_edge(n1, n2);

    SUBCASE("add_node") {
      Node n3 = g.add_node();
      std::unordered_set<Node> nodes = g.query_nodes(NodeQuery{{n3}});
      CHECK(contains(nodes, n3));
    }

    SUBCASE("add_edge") {
      MultiDiEdge e4 = g.add_edge(n2, n1);
      std::unordered_set<MultiDiEdge> edges =
          g.query_edges(MultiDiEdgeQuery({n2}, {n1}));
      CHECK(contains(edges, e4));
    }

    SUBCASE("remove_node") {
      g.remove_node(n0);
      std::unordered_set<Node> nodes = g.query_nodes(NodeQuery{{n0}});
      CHECK_FALSE(contains(nodes, n0));

      std::unordered_set<MultiDiEdge> edges_after_removal =
          g.query_edges(MultiDiEdgeQuery({n0}, {n0}));
      CHECK_FALSE(contains(edges_after_removal, e0));
      CHECK_FALSE(contains(edges_after_removal, e1));
      CHECK_FALSE(contains(edges_after_removal, e2));
    }

    SUBCASE("remove_edge") {
      g.remove_edge(e3);
      std::unordered_set<MultiDiEdge> edges =
          g.query_edges(MultiDiEdgeQuery({n1}, {n2}));
      CHECK_FALSE(contains(edges, e3));
    }

    SUBCASE("query_nodes") {
      std::unordered_set<Node> all_nodes =
          g.query_nodes(NodeQuery{{n0, n1, n2}});
      CHECK(all_nodes == std::unordered_set<Node>{n0, n1, n2});

      std::unordered_set<Node> specific_nodes =
          g.query_nodes(NodeQuery{{n0, n2}});
      CHECK(specific_nodes == std::unordered_set<Node>{n0, n2});
    }

    SUBCASE("query_edges") {
      std::unordered_set<MultiDiEdge> all_edges =
          g.query_edges(MultiDiEdgeQuery({n0, n1, n2}, {n0, n1, n2}));
      CHECK(all_edges == std::unordered_set<MultiDiEdge>{e0, e1, e2, e3});

      std::unordered_set<MultiDiEdge> edges_from_n1 =
          g.query_edges(MultiDiEdgeQuery({n1}, {n0, n1, n2}));
      CHECK(edges_from_n1 == std::unordered_set<MultiDiEdge>{e0, e3});

      std::unordered_set<MultiDiEdge> edges_to_n2 =
          g.query_edges(MultiDiEdgeQuery({n0, n1, n2}, {n2}));
      CHECK(edges_to_n2 == std::unordered_set<MultiDiEdge>{e2, e3});
    }

    SUBCASE("get_multidiedge_src") {
      CHECK(g.get_multidiedge_src(e0) == n1);
      CHECK(g.get_multidiedge_dst(e0) == n0);
    }

    SUBCASE("get_multidiedge_dst") {
      CHECK(g.get_multidiedge_src(e2) == n0);
      CHECK(g.get_multidiedge_dst(e2) == n2);
    }
  }
}
