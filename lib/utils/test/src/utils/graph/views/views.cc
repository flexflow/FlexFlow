#include "utils/graph/views/views.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/fmt/unordered_map.h"
#include "utils/fmt/unordered_set.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/instances/hashmap_undirected_graph.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/undirected/undirected_graph.h"
#include "utils/graph/undirected/undirected_graph_view.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("UndirectedSubgraphView") {
    UndirectedGraph g = UndirectedGraph::create<HashmapUndirectedGraph>();
    std::vector<Node> n = add_nodes(g, 5);
    add_edges(g,
              {{n[0], n[3]},
               {n[1], n[1]},
               {n[1], n[2]},
               {n[1], n[3]},
               {n[2], n[3]},
               {n[2], n[4]}});
    std::unordered_set<Node> sub_nodes = {n[0], n[1], n[3]};
    UndirectedGraphView view =
        UndirectedGraphView::create<UndirectedSubgraphView>(g, sub_nodes);

    SUBCASE("get_nodes") {
      std::unordered_set<Node> expected = {n[0], n[1], n[3]};

      std::unordered_set<Node> result = get_nodes(view);

      CHECK(result == expected);
    }

    SUBCASE("get_edges") {
      std::unordered_set<UndirectedEdge> expected = {
          {n[0], n[3]},
          {n[1], n[1]},
          {n[1], n[3]},
      };

      std::unordered_set<UndirectedEdge> result = get_edges(view);

      // CHECK(result == expected);
    }
  }

  TEST_CASE("DiSubgraphView") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    std::vector<Node> n = add_nodes(g, 5);
    add_edges(g,
              {DirectedEdge{n[0], n[3]},
               DirectedEdge{n[3], n[0]},
               DirectedEdge{n[1], n[1]},
               DirectedEdge{n[2], n[1]},
               DirectedEdge{n[1], n[3]},
               DirectedEdge{n[2], n[3]},
               DirectedEdge{n[3], n[2]},
               DirectedEdge{n[2], n[4]}});
    std::unordered_set<Node> sub_nodes = {n[0], n[1], n[3]};
    DiGraphView view = DiGraphView::create<DiSubgraphView>(g, sub_nodes);

    SUBCASE("get_nodes") {
      std::unordered_set<Node> expected = {n[0], n[1], n[3]};

      std::unordered_set<Node> result = get_nodes(view);

      CHECK(result == expected);
    }

    SUBCASE("get_edges") {
      std::unordered_set<DirectedEdge> expected = {
          DirectedEdge{n[0], n[3]},
          DirectedEdge{n[3], n[0]},
          DirectedEdge{n[1], n[1]},
          DirectedEdge{n[1], n[3]},
      };

      std::unordered_set<DirectedEdge> result = get_edges(view);

      CHECK(result == expected);
    }
  }

  // TEST_CASE("JoinedUndirectedGraphView") {
  //       UndirectedGraph g1 =
  //       UndirectedGraph::create<HashmapUndirectedGraph>(); UndirectedGraph g2
  //       = UndirectedGraph::create<HashmapUndirectedGraph>();

  //       std::vector<Node> n1 = add_nodes(g1, 3);
  //       std::vector<Node> n2 = add_nodes(g2, 3);

  //       add_edges(g1, {{n1[0], n1[1]}, {n1[1], n1[2]}});
  //       add_edges(g2, {{n2[0], n2[2]}, {n2[1], n2[2]}});

  //       UndirectedGraphView view =
  //       UndirectedGraphView::create<JoinedUndirectedGraphView>(g1, g2);

  //       SUBCASE("get_nodes") {
  //           std::unordered_set<Node> expected =
  //           set_union(unordered_set_of(n1), unordered_set_of(n2));

  //           std::unordered_set<Node> result = get_nodes(view);

  //           CHECK(result == expected);
  //       }

  //       SUBCASE("get_edges") {
  //           std::unordered_set<UndirectedEdge> expected = {
  //               {n1[0], n1[1]}, {n1[1], n1[2]},
  //               {n2[0], n2[2]}, {n2[1], n2[2]}
  //           };

  //           std::unordered_set<UndirectedEdge> result = get_edges(view);

  //           CHECK(result == expected);
  //       }
  //   }

  // TEST_CASE("JoinedDigraphView") {
  //     DiGraph g1 = DiGraph::create<AdjacencyDiGraph>();
  //     DiGraph g2 = DiGraph::create<AdjacencyDiGraph>();

  //     std::vector<Node> n1 = add_nodes(g1, 3);
  //     std::vector<Node> n2 = add_nodes(g2, 3);

  //     add_edges(g1, {DirectedEdge{n1[0], n1[1]}, DirectedEdge{n1[1],
  //     n1[2]}}); add_edges(g2, {DirectedEdge{n2[0], n2[2]},
  //     DirectedEdge{n2[1], n2[2]}});

  //     DiGraphView view = DiGraphView::create<JoinedDigraphView>(g1, g2);

  //     SUBCASE("get_nodes") {
  //         std::unordered_set<Node> expected = set_union(unordered_set_of(n1),
  //         unordered_set_of(n2));

  //         std::unordered_set<Node> result = get_nodes(view);

  //         CHECK(result == expected);
  //     }

  //     SUBCASE("get_edges") {
  //         std::unordered_set<DirectedEdge> expected = {
  //             DirectedEdge{n1[0], n1[1]}, DirectedEdge{n1[1], n1[2]},
  //             DirectedEdge{n2[0], n2[2]}, DirectedEdge{n2[1], n2[2]}
  //         };

  //         std::unordered_set<DirectedEdge> result = get_edges(view);

  //         CHECK(result == expected);
  //     }
  // }

  // TEST_CASE("AddDirectedEdgesView") {
  //     DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  //     std::vector<Node> n = add_nodes(g, 4);
  //     add_edges(g, {DirectedEdge{n[0], n[1]}, DirectedEdge{n[1], n[2]}});

  //     std::unordered_set<DirectedEdge> additional_edges = {
  //         DirectedEdge{n[2], n[3]}, DirectedEdge{n[3], n[0]}
  //     };

  //     DiGraphView view = DiGraphView::create<AddDirectedEdgesView>(g,
  //     additional_edges);

  //     SUBCASE("get_nodes") {
  //         std::unordered_set<Node> expected = unordered_set_of(n);

  //         std::unordered_set<Node> result = get_nodes(view);

  //         CHECK(result == expected);
  //     }

  //     SUBCASE("get_edges") {
  //         std::unordered_set<DirectedEdge> expected = {
  //             DirectedEdge{n[0], n[1]}, DirectedEdge{n[1], n[2]},
  //             DirectedEdge{n[2], n[3]}, DirectedEdge{n[3], n[0]}
  //         };

  //         std::unordered_set<DirectedEdge> result = get_edges(view);

  //         CHECK(result == expected);
  //     }
  // }

  TEST_CASE("ViewDiGraphAsUndirectedGraph") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    std::vector<Node> n = add_nodes(g, 3);
    add_edges(g,
              {DirectedEdge{n[0], n[1]},
               DirectedEdge{n[1], n[2]},
               DirectedEdge{n[2], n[0]},
               DirectedEdge{n[0], n[2]}});

    UndirectedGraphView view =
        UndirectedGraphView::create<ViewDiGraphAsUndirectedGraph>(g);

    SUBCASE("get_nodes") {
      std::unordered_set<Node> expected = unordered_set_of(n);

      std::unordered_set<Node> result = get_nodes(view);

      CHECK(result == expected);
    }

    SUBCASE("get_edges") {
      std::unordered_set<UndirectedEdge> expected = {
          {n[0], n[1]}, {n[1], n[2]}, {n[2], n[0]}};

      std::unordered_set<UndirectedEdge> result = get_edges(view);

      CHECK(result == expected);
    }
  }

  TEST_CASE("ViewUndirectedGraphAsDiGraph") {
    UndirectedGraph g = UndirectedGraph::create<HashmapUndirectedGraph>();
    std::vector<Node> n = add_nodes(g, 3);
    add_edges(g, {{n[0], n[0]}, {n[0], n[1]}, {n[1], n[2]}, {n[2], n[0]}});

    DiGraphView view = DiGraphView::create<ViewUndirectedGraphAsDiGraph>(g);

    SUBCASE("get_nodes") {
      std::unordered_set<Node> expected = unordered_set_of(n);

      std::unordered_set<Node> result = get_nodes(view);

      CHECK(result == expected);
    }

    SUBCASE("get_edges") {
      std::unordered_set<DirectedEdge> expected = {DirectedEdge{n[0], n[0]},
                                                   DirectedEdge{n[0], n[1]},
                                                   DirectedEdge{n[1], n[0]},
                                                   DirectedEdge{n[1], n[2]},
                                                   DirectedEdge{n[2], n[1]},
                                                   DirectedEdge{n[2], n[0]},
                                                   DirectedEdge{n[0], n[2]}};

      std::unordered_set<DirectedEdge> result = get_edges(view);

      CHECK(result == expected);
    }
  }
}
