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
              {UndirectedEdge{{n.at(0), n.at(3)}},
               UndirectedEdge{{n.at(1), n.at(1)}},
               UndirectedEdge{{n.at(1), n.at(2)}},
               UndirectedEdge{{n.at(1), n.at(3)}},
               UndirectedEdge{{n.at(2), n.at(3)}},
               UndirectedEdge{{n.at(2), n.at(4)}}});
    std::unordered_set<Node> sub_nodes = {n.at(0), n.at(1), n.at(3)};
    UndirectedGraphView view = get_subgraph(g, sub_nodes);

    SUBCASE("get_nodes") {
      std::unordered_set<Node> expected = {n.at(0), n.at(1), n.at(3)};

      std::unordered_set<Node> result = get_nodes(view);

      CHECK(result == expected);
    }

    SUBCASE("get_edges") {
      std::unordered_set<UndirectedEdge> expected = {
          UndirectedEdge{{n.at(0), n.at(3)}},
          UndirectedEdge{{n.at(1), n.at(1)}},
          UndirectedEdge{{n.at(1), n.at(3)}},
      };

      std::unordered_set<UndirectedEdge> result = get_edges(view);

      // TODO(@pietro) TODO(@lockshaw) current BUG, get_edges also
      CHECK(result == expected);
    }
  }

  TEST_CASE("DiSubgraphView") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    std::vector<Node> n = add_nodes(g, 5);
    add_edges(g,
              {DirectedEdge{n.at(0), n.at(3)},
               DirectedEdge{n.at(3), n.at(0)},
               DirectedEdge{n.at(1), n.at(1)},
               DirectedEdge{n.at(2), n.at(1)},
               DirectedEdge{n.at(1), n.at(3)},
               DirectedEdge{n.at(2), n.at(3)},
               DirectedEdge{n.at(3), n.at(2)},
               DirectedEdge{n.at(2), n.at(4)}});
    std::unordered_set<Node> sub_nodes = {n.at(0), n.at(1), n.at(3)};
    DiGraphView view = get_subgraph(g, sub_nodes);

    SUBCASE("get_nodes") {
      std::unordered_set<Node> expected = {n.at(0), n.at(1), n.at(3)};

      std::unordered_set<Node> result = get_nodes(view);

      CHECK(result == expected);
    }

    SUBCASE("get_edges") {
      std::unordered_set<DirectedEdge> expected = {
          DirectedEdge{n.at(0), n.at(3)},
          DirectedEdge{n.at(3), n.at(0)},
          DirectedEdge{n.at(1), n.at(1)},
          DirectedEdge{n.at(1), n.at(3)},
      };

      std::unordered_set<DirectedEdge> result = get_edges(view);

      CHECK(result == expected);
    }
  }

  TEST_CASE("JoinedUndirectedGraphView") {
    UndirectedGraph g1 = UndirectedGraph::create<HashmapUndirectedGraph>();
    UndirectedGraph g2 = UndirectedGraph::create<HashmapUndirectedGraph>();

    std::vector<Node> n1 = add_nodes(g1, 3);
    std::vector<Node> n2 = add_nodes(g2, 3);

    add_edges(g1,
              {UndirectedEdge{{n1.at(0), n1.at(1)}},
               UndirectedEdge{{n1.at(1), n1.at(2)}}});
    add_edges(g2,
              {UndirectedEdge{{n2.at(0), n2.at(2)}},
               UndirectedEdge{{n2.at(1), n2.at(2)}}});

    UndirectedGraphView view = join(g1, g2);

    SUBCASE("get_nodes") {
      std::unordered_set<Node> expected =
          set_union(unordered_set_of(n1), unordered_set_of(n2));

      std::unordered_set<Node> result = get_nodes(view);

      CHECK(result == expected);
    }

    // SUBCASE("get_edges") {
    //     std::unordered_set<UndirectedEdge> expected = {
    //         UndirectedEdge{{n1.at(0), n1.at(1)}}, UndirectedEdge{{n1.at(1),
    //         n1.at(2)}}, UndirectedEdge{{n2.at(0), n2.at(2)}},
    //         UndirectedEdge{{n2.at(1), n2.at(2)}}
    //     };

    //     std::unordered_set<UndirectedEdge> result = get_edges(view);

    //     CHECK(result == expected);
    // }
  }

  TEST_CASE("JoinedDigraphView") {
    DiGraph g1 = DiGraph::create<AdjacencyDiGraph>();
    DiGraph g2 = DiGraph::create<AdjacencyDiGraph>();

    std::vector<Node> n1 = add_nodes(g1, 3);
    std::vector<Node> n2 = add_nodes(g2, 3);

    add_edges(
        g1,
        {DirectedEdge{n1.at(0), n1.at(1)}, DirectedEdge{n1.at(1), n1.at(2)}});
    add_edges(
        g2,
        {DirectedEdge{n2.at(0), n2.at(2)}, DirectedEdge{n2.at(1), n2.at(2)}});

    DiGraphView view = join(g1, g2);

    SUBCASE("get_nodes") {
      std::unordered_set<Node> expected =
          set_union(unordered_set_of(n1), unordered_set_of(n2));

      std::unordered_set<Node> result = get_nodes(view);

      CHECK(result == expected);
    }

    //     SUBCASE("get_edges") {
    //         std::unordered_set<DirectedEdge> expected = {
    //             DirectedEdge{n1.at(0), n1.at(1)}, DirectedEdge{n1.at(1),
    //             n1.at(2)}, DirectedEdge{n2.at(0), n2.at(2)},
    //             DirectedEdge{n2.at(1), n2.at(2)}
    //         };

    //         std::unordered_set<DirectedEdge> result = get_edges(view);

    //         CHECK(result == expected);
    //     }
  }

  TEST_CASE("ViewDiGraphAsUndirectedGraph") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    std::vector<Node> n = add_nodes(g, 3);
    add_edges(g,
              {DirectedEdge{n.at(0), n.at(1)},
               DirectedEdge{n.at(1), n.at(2)},
               DirectedEdge{n.at(2), n.at(0)},
               DirectedEdge{n.at(0), n.at(2)}});

    UndirectedGraphView view = as_undirected(g);

    SUBCASE("get_nodes") {
      std::unordered_set<Node> expected = unordered_set_of(n);

      std::unordered_set<Node> result = get_nodes(view);

      CHECK(result == expected);
    }

    SUBCASE("get_edges") {
      std::unordered_set<UndirectedEdge> expected = {
          UndirectedEdge{{n.at(0), n.at(1)}},
          UndirectedEdge{{n.at(1), n.at(2)}},
          UndirectedEdge{{n.at(2), n.at(0)}}};

      std::unordered_set<UndirectedEdge> result = get_edges(view);

      CHECK(result == expected);
    }
  }

  TEST_CASE("ViewUndirectedGraphAsDiGraph") {
    UndirectedGraph g = UndirectedGraph::create<HashmapUndirectedGraph>();
    std::vector<Node> n = add_nodes(g, 3);
    add_edges(g,
              {UndirectedEdge{{n.at(0), n.at(0)}},
               UndirectedEdge{{n.at(0), n.at(1)}},
               UndirectedEdge{{n.at(1), n.at(2)}},
               UndirectedEdge{{n.at(2), n.at(0)}}});

    DiGraphView view = as_digraph(g);

    SUBCASE("get_nodes") {
      std::unordered_set<Node> expected = unordered_set_of(n);

      std::unordered_set<Node> result = get_nodes(view);

      CHECK(result == expected);
    }

    SUBCASE("get_edges") {
      std::unordered_set<DirectedEdge> expected = {
          DirectedEdge{n.at(0), n.at(0)},
          DirectedEdge{n.at(0), n.at(1)},
          DirectedEdge{n.at(1), n.at(0)},
          DirectedEdge{n.at(1), n.at(2)},
          DirectedEdge{n.at(2), n.at(1)},
          DirectedEdge{n.at(2), n.at(0)},
          DirectedEdge{n.at(0), n.at(2)}};

      std::unordered_set<DirectedEdge> result = get_edges(view);

      CHECK(result == expected);
    }
  }
}
