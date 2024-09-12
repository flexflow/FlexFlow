#include "utils/graph/undirected/algorithms/get_connected_components.h"
#include "test/utils/doctest.h"
#include "utils/fmt/unordered_set.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/instances/hashmap_undirected_graph.h"
#include "utils/graph/undirected/undirected_graph.h"

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("get_connected_components") {
    UndirectedGraph g = UndirectedGraph::create<HashmapUndirectedGraph>();
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
}
