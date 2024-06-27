#include "test/utils/doctest.h"
#include "utils/graph/adjacency_digraph.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph.h"
#include "utils/graph/serialparallel.h"
#include "utils/graph/simple_sp_ization.h"
using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("Simple SP-ization Algorithm") {
    /*
    digraph G {
    n0 [label="n0\nlayer=0"];
    n1 [label="n1\nlayer=1"];
    n2 [label="n2\nlayer=1"];
    n3 [label="n3\nlayer=2"];
    n4 [label="n4\nlayer=3"];
    n5 [label="n5\nlayer=4"];
    n6 [label="n6\nlayer=2"];

    n0 -> n1;
    n0 -> n2;
    n2 -> n3;
    n1 -> n4;
    n3 -> n4;
    n3 -> n5;
    n4 -> n5;
    n0 -> n6;
    n2 -> n6;
    n6 -> n5;
}

    */
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    std::vector<Node> n = add_nodes(g, 7);

    // Currently doesn't work due to DiEdge flipping graph
    g.add_edge({n[0], n[1]});
    g.add_edge({n[0], n[2]});
    g.add_edge({n[2], n[3]});
    g.add_edge({n[1], n[4]});
    g.add_edge({n[3], n[4]});
    g.add_edge({n[3], n[5]});
    g.add_edge({n[4], n[5]});
    g.add_edge({n[0], n[6]});
    g.add_edge({n[2], n[6]});
    g.add_edge({n[6], n[5]});

    SerialParallelDecomposition sp = simple_sp_ization(g);
    CHECK(std::holds_alternative<Serial>(sp));
    Serial sp_root = std::get<Serial>(sp);
    CHECK(sp_root.children.size() == 5);
    std::vector<int> sizes = {1, 2, 2, 1, 1};
    size_t i = 0;

    for (auto const &node : sp_root.children) {
      CHECK(std::holds_alternative<Parallel>(node));
      Parallel par_node = std::get<Parallel>(node);
      CHECK(par_node.children.size() == sizes[i]);
      i++;
    }
  }
}
