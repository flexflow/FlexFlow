#include "utils/graph/digraph/algorithms/get_topological_ordering.h"
#include "test/utils/doctest.h"
#include "utils/containers/index_of.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/digraph.h"
#include "utils/graph/instances/adjacency_digraph.h"

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("get_topological_ordering") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    std::vector<Node> n = add_nodes(g, 6);
    std::vector<DirectedEdge> edges = {DirectedEdge{n.at(0), n.at(1)},
                                       DirectedEdge{n.at(0), n.at(2)},
                                       DirectedEdge{n.at(1), n.at(5)},
                                       DirectedEdge{n.at(2), n.at(3)},
                                       DirectedEdge{n.at(3), n.at(4)},
                                       DirectedEdge{n.at(4), n.at(5)}};
    add_edges(g, edges);
    std::vector<Node> ordering = get_topological_ordering(g);
    auto CHECK_BEFORE = [&](int l, int r) {
      CHECK(index_of(ordering, n[l]).value() <
            index_of(ordering, n[r]).value());
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
