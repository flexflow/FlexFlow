#include "utils/graph/views/views.h"
#include "utils/containers/set_union.h"
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
    UndirectedGraphView view = view_subgraph(g, sub_nodes);

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
    DiGraphView view = view_subgraph(g, sub_nodes);

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
