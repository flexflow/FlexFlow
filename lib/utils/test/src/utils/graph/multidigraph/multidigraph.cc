#include "utils/graph/multidigraph/multidigraph.h"
#include "utils/containers/contains.h"
#include "utils/graph/instances/adjacency_multidigraph.h"
#include "utils/graph/multidigraph/multidiedge_query.h"
#include "utils/graph/query_set.h"
#include <doctest/doctest.h>
#include <unordered_set>
#include <vector>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("MultiDiGraph") {
    MultiDiGraph g = MultiDiGraph::create<AdjacencyMultiDiGraph>();

    Node n0 = g.add_node();
    Node n1 = g.add_node();
    Node n2 = g.add_node();
    MultiDiEdge e0 = g.add_edge(n0, n2);
    MultiDiEdge e1 = g.add_edge(n1, n0);
    MultiDiEdge e2 = g.add_edge(n1, n0);
    MultiDiEdge e3 = g.add_edge(n1, n2);
    MultiDiEdge e4 = g.add_edge(n1, n2);
    MultiDiEdge e5 = g.add_edge(n2, n0);
    MultiDiEdge e6 = g.add_edge(n2, n2);

    SUBCASE("add_node") {
      Node n3 = g.add_node();
      std::unordered_set<Node> result = g.query_nodes(NodeQuery{{n3}});
      std::unordered_set<Node> correct = {n3};
      CHECK(result == correct);
    }

    SUBCASE("add_edge") {
      SUBCASE("non-duplicate edge") {
        MultiDiEdge e7 = g.add_edge(n2, n1);
        std::unordered_set<MultiDiEdge> result =
            g.query_edges(MultiDiEdgeQuery({n2}, {n1}));
        std::unordered_set<MultiDiEdge> correct = {e7};
        CHECK(result == correct);
      }

      SUBCASE("duplicate edge") {
        MultiDiEdge e7 = g.add_edge(n2, n1);
        MultiDiEdge e8 = g.add_edge(n2, n1);

        std::unordered_set<MultiDiEdge> result =
            g.query_edges(MultiDiEdgeQuery({n2}, {n1}));
        std::unordered_set<MultiDiEdge> correct = {e7, e8};
        CHECK(result == correct);
      }
    }

    SUBCASE("remove_node") {
      g.remove_node(n0);

      std::unordered_set<Node> node_result = g.query_nodes(NodeQuery{{n0}});
      std::unordered_set<Node> node_correct = {};
      CHECK(node_result == node_correct);

      std::unordered_set<MultiDiEdge> edge_result =
          g.query_edges(MultiDiEdgeQuery({n0}, {n1, n2}));
      std::unordered_set<MultiDiEdge> edge_correct = {};
      CHECK(edge_result == edge_correct);
    }

    SUBCASE("remove_edge") {
      g.remove_edge(e3);
      std::unordered_set<MultiDiEdge> result =
          g.query_edges(MultiDiEdgeQuery({n1}, {n2}));
      std::unordered_set<MultiDiEdge> correct = {e4};
      CHECK(result == correct);

      SUBCASE("remove non-duplicate edge") {
        g.remove_edge(e0);
        std::unordered_set<MultiDiEdge> result =
            g.query_edges(MultiDiEdgeQuery({n0}, {n2}));
        std::unordered_set<MultiDiEdge> correct = {};
        CHECK(result == correct);
      }

      SUBCASE("remove duplicate edge") {
        g.remove_edge(e1);
        std::unordered_set<MultiDiEdge> result =
            g.query_edges(MultiDiEdgeQuery({n1}, {n0}));
        std::unordered_set<MultiDiEdge> correct = {e2};
        CHECK(result == correct);
      }
    }

    SUBCASE("query_nodes") {
      SUBCASE("all nodes") {
        std::unordered_set<Node> result =
            g.query_nodes(NodeQuery{{n0, n1, n2}});
        std::unordered_set<Node> correct = {n0, n1, n2};
        CHECK(result == correct);
      }

      SUBCASE("specific nodes") {
        std::unordered_set<Node> result = g.query_nodes(NodeQuery{{n0, n2}});
        std::unordered_set<Node> correct = {n0, n2};
        CHECK(result == correct);
      }

      SUBCASE("matchall") {
        std::unordered_set<Node> result =
            g.query_nodes(NodeQuery{matchall<Node>()});
        std::unordered_set<Node> correct = {n0, n1, n2};
        CHECK(result == correct);
      }

      SUBCASE("nodes not in graph") {
        Node n3 = Node(3);
        Node n4 = Node(4);
        std::unordered_set<Node> result = g.query_nodes(NodeQuery{{n3, n4}});
        std::unordered_set<Node> correct = {};
        CHECK(result == correct);
      }
    }

    SUBCASE("query_edges") {
      SUBCASE("all edges") {
        std::unordered_set<MultiDiEdge> result =
            g.query_edges(MultiDiEdgeQuery({n0, n1, n2}, {n0, n1, n2}));
        std::unordered_set<MultiDiEdge> correct = {e0, e1, e2, e3, e4, e5, e6};
        CHECK(result == correct);
      }

      SUBCASE("edges from n1") {
        std::unordered_set<MultiDiEdge> result =
            g.query_edges(MultiDiEdgeQuery({n1}, {n0, n1, n2}));
        std::unordered_set<MultiDiEdge> correct = {e1, e2, e3, e4};
        CHECK(result == correct);
      }

      SUBCASE("edges to n2") {
        std::unordered_set<MultiDiEdge> result =
            g.query_edges(MultiDiEdgeQuery({n0, n1, n2}, {n2}));
        std::unordered_set<MultiDiEdge> correct = {e0, e3, e4, e6};
        CHECK(result == correct);
      }

      SUBCASE("matchall") {
        std::unordered_set<MultiDiEdge> result =
            g.query_edges(MultiDiEdgeQuery(matchall<Node>(), matchall<Node>()));
        std::unordered_set<MultiDiEdge> correct = {e0, e1, e2, e3, e4, e5, e6};
        CHECK(result == correct);
      }

      SUBCASE("nodes that don't exist") {
        Node n3 = Node(3);
        Node n4 = Node(4);
        std::unordered_set<MultiDiEdge> result =
            g.query_edges(MultiDiEdgeQuery({n1, n3}, {n4}));
        std::unordered_set<MultiDiEdge> correct = {};
        CHECK(result == correct);
      }
    }
    SUBCASE("get_multidiedge_src") {
      Node result = g.get_multidiedge_src(e0);
      Node correct = n0;
      CHECK(result == correct);
    }

    SUBCASE("get_multidiedge_dst") {
      Node result = g.get_multidiedge_dst(e0);
      Node correct = n2;
      CHECK(result == correct);
    }
  }
}
