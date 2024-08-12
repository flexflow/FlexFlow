#include "utils/graph/digraph/algorithms/get_topological_ordering.h"
#include "test/utils/doctest.h"
#include "utils/containers.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/digraph.h"
#include "utils/graph/instances/adjacency_digraph.h"

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("get_topological_ordering") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    std::vector<Node> n = add_nodes(g, 6);
    std::vector<DirectedEdge> edges = {DirectedEdge{n[0], n[1]},
                                       DirectedEdge{n[0], n[2]},
                                       DirectedEdge{n[1], n[5]},
                                       DirectedEdge{n[2], n[3]},
                                       DirectedEdge{n[3], n[4]},
                                       DirectedEdge{n[4], n[5]}};
    add_edges(g, edges);
    std::vector<Node> ordering = get_topological_ordering(g);
    auto CHECK_BEFORE = [&](int l, int r) {
      CHECK(index_of(ordering, n[l]).has_value());
      CHECK(index_of(ordering, n[r]).has_value());
      CHECK(index_of(ordering, n[l]) < index_of(ordering, n[r]));
    };

    CHECK(ordering.size() == n.size());
    CHECK_BEFORE(0, 1);
    CHECK_BEFORE(0, 2);
    CHECK_BEFORE(1, 5);
    CHECK_BEFORE(2, 3);
    CHECK_BEFORE(3, 4);
    CHECK_BEFORE(4, 5);
  }
}
