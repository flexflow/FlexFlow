#include "utils/graph/undirected/algorithms/get_connected_components.h"
#include "utils/fmt/unordered_set.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/instances/hashmap_undirected_graph.h"
#include "utils/graph/undirected/undirected_graph.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_CASE("get_connected_components") {
  UndirectedGraph g = UndirectedGraph::create<HashmapUndirectedGraph>();

  SUBCASE("disjoint nodes") {
    std::vector<Node> n = add_nodes(g, 3);

    std::unordered_set<std::unordered_set<Node>> correct = {
        {n[0]},
        {n[1]},
        {n[2]},
    };
    std::unordered_set<std::unordered_set<Node>> result =
        get_connected_components(g);

    CHECK(correct == result);
  }

  SUBCASE("2 components") {
    std::vector<Node> n = add_nodes(g, 4);
    add_edges(g, {UndirectedEdge{{n[0], n[1]}}, UndirectedEdge{{n[2], n[1]}}});

    std::unordered_set<std::unordered_set<Node>> correct = {
        {n[0], n[1], n[2]},
        {n[3]},
    };
    std::unordered_set<std::unordered_set<Node>> result =
        get_connected_components(g);

    CHECK(correct == result);
  }

  SUBCASE("3 components") {
    std::vector<Node> n = add_nodes(g, 6);
    add_edges(g,
              {
                  UndirectedEdge{{n[0], n[1]}},
                  UndirectedEdge{{n[0], n[2]}},
                  UndirectedEdge{{n[1], n[2]}},
                  UndirectedEdge{{n[3], n[4]}},
              });

    std::unordered_set<std::unordered_set<Node>> correct = {
        {n[0], n[1], n[2]},
        {n[3], n[4]},
        {n[5]},
    };
    std::unordered_set<std::unordered_set<Node>> result =
        get_connected_components(g);

    CHECK(correct == result);
  }
}
