#include "utils/graph/undirected/algorithms/get_connected_components.h"
#include "test/utils/doctest.h"
#include "utils/graph/undirected/undirected_graph.h"
#include "utils/graph/instances/hashmap_undirected_graph.h"
#include "utils/graph/algorithms.h"

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("get_connected_components") {
    UndirectedGraph g = UndirectedGraph::create<HashmapUndirectedGraph>();
    std::vector<Node> n = add_nodes(g, 4);
    std::vector<UndirectedEdge> edges = {{n[0], n[1]}, {n[2], n[1]}};

    add_edges(g, edges);
    std::unordered_set<std::unordered_set<Node>> expected_components = {
        {n[0], n[1], n[2]},
        {n[3]},
    };

    CHECK(get_connected_components(g) == expected_components);
  }
}
