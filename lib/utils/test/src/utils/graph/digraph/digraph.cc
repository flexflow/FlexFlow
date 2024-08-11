#include "utils/graph/digraph/digraph.h"
#include "utils/containers/repeat.h"
#include "utils/graph/digraph/directed_edge_query.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/node/node_query.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE_TEMPLATE("DiGraph implementations", T, AdjacencyDiGraph) {
    /*
    graph TD

    n0 --> n1
    n0 --> n2
    n1 --> n2
    n2 --> n4
    n1 --> n3
    */

    DiGraph g = DiGraph::create<T>();
    std::vector<Node> n = repeat(5, [&] { return g.add_node(); });
    std::vector<DirectedEdge> e = {DirectedEdge{n[0], n[1]},
                                   DirectedEdge{n[0], n[2]},
                                   DirectedEdge{n[1], n[2]},
                                   DirectedEdge{n[2], n[4]},
                                   DirectedEdge{n[1], n[3]}};
    for (DirectedEdge const &edge : e) {
      g.add_edge(edge);
    }

    SUBCASE("query_nodes") {

      CHECK(g.query_nodes(node_query_all()) ==
            std::unordered_set<Node>{n[0], n[1], n[2], n[3], n[4]});

      CHECK(g.query_nodes(NodeQuery{query_set<Node>{{n[0], n[2]}}}) ==
            std::unordered_set<Node>{n[0], n[2]});

      std::unordered_set<DirectedEdge> queried_edges =
          g.query_edges(directed_edge_query_all());
      std::unordered_set<DirectedEdge> expected = {
          e[0], e[1], e[2], e[3], e[4]};
      CHECK(queried_edges == expected);

      queried_edges = g.query_edges(
          DirectedEdgeQuery{query_set<Node>{{n[0]}}, query_set<Node>{{n[1]}}});
      expected = std::unordered_set<DirectedEdge>{e[0]};
      CHECK(queried_edges == expected);
    }
    SUBCASE("remove_node_unsafe") {
      g.remove_node_unsafe(n[0]);

      CHECK(g.query_nodes(node_query_all()) ==
            std::unordered_set<Node>{n[1], n[2], n[3], n[4]});

      // removing a node also removes its adjacent edges
      CHECK(g.query_edges(directed_edge_query_all()) ==
            std::unordered_set<DirectedEdge>{e[2], e[3], e[4]});

      g.remove_node_unsafe(n[1]);

      CHECK(g.query_nodes(node_query_all()) ==
            std::unordered_set<Node>{n[2], n[3], n[4]});

      CHECK(g.query_edges(directed_edge_query_all()) ==
            std::unordered_set<DirectedEdge>{e[3]});
    }

    SUBCASE("remove_edge") {
      g.remove_edge(e[0]);

      CHECK(g.query_edges(directed_edge_query_all()) ==
            std::unordered_set<DirectedEdge>{e[1], e[2], e[3], e[4]});
      CHECK(g.query_nodes(node_query_all()) ==
            std::unordered_set<Node>{n[0], n[1], n[2], n[3], n[4]});

      g.remove_edge(e[1]);
      g.remove_edge(e[3]);
      CHECK(g.query_edges(directed_edge_query_all()) ==
            std::unordered_set<DirectedEdge>{e[2], e[4]});
    }
  }
}
