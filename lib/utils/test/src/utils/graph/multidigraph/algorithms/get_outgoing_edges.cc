#include "utils/graph/multidigraph/algorithms/get_outgoing_edges.h"
#include "test/utils/doctest.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/instances/adjacency_multidigraph.h"
#include "utils/graph/multidigraph/algorithms/add_edges.h"
#include "utils/graph/multidigraph/algorithms/add_nodes.h"
#include "utils/graph/multidigraph/multidigraph.h"
#include <unordered_set>
#include <vector>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("MultiDiGraph - get_outgoing_edges") {
    MultiDiGraph g = MultiDiGraph::create<AdjacencyMultiDiGraph>();
    std::vector<Node> n = add_nodes(g, 3);

    std::vector<std::pair<Node, Node>> input = {
        {n.at(0), n.at(1)},
        {n.at(0), n.at(1)},
        {n.at(1), n.at(1)},
        {n.at(0), n.at(0)},
    };

    std::vector<MultiDiEdge> edges = add_edges(g, input);

    CHECK(get_outgoing_edges(g, n[0]) ==
          std::unordered_set<MultiDiEdge>{edges[0], edges[1], edges[3]});
    CHECK(get_outgoing_edges(g, n[2]) == std::unordered_set<MultiDiEdge>{});
  }
}
