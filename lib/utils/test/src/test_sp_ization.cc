#include "test/utils/doctest.h"
#include "utils/graph/adjacency_digraph.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph.h"
#include "utils/graph/serialparallel.h"
#include "utils/graph/sp_ization.h"

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("Barrier Syncing SP-ization Algorithm") {
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

    SerialParallelDecomposition sp = barrier_sync_sp_ization(g);
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

  TEST_CASE("Dependency Invariant SP-ization algorithm - Straight Line") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    std::vector<Node> n = add_nodes(g, 4);
    g.add_edge({n[0], n[1]});
    g.add_edge({n[1], n[2]});
    g.add_edge({n[2], n[3]});

    auto gv = flipped(g); // flipped to account for the diedge bug
    SerialParallelDecomposition result =
        dependency_invariant_sp_ization(gv);
    SerialParallelDecomposition expected = Serial{{n[0], n[1], n[2], n[3]}};
    CHECK(
        std::get<Serial>(result) ==
        std::get<Serial>(expected)); // currently cannot directly compare the 2.

    result =
        dependency_invariant_sp_ization_with_coalescing(gv);
    CHECK(
        std::get<Serial>(result) ==
        std::get<Serial>(expected)); // currently cannot directly compare the 2.
  }

  TEST_CASE("Dependency Invariant SP-ization algorithm - Rhombus Pattern") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    std::vector<Node> n = add_nodes(g, 4);
    g.add_edge({n[0], n[1]});
    g.add_edge({n[0], n[2]});
    g.add_edge({n[1], n[3]});
    g.add_edge({n[2], n[3]});

    auto gv = flipped(g); // flipped to account for the diedge bug
    SerialParallelDecomposition result =
        dependency_invariant_sp_ization(gv);
    SerialParallelDecomposition expected = Serial{{n[0], Parallel{{n[2], n[1]}}, n[3]}};

    result =
        dependency_invariant_sp_ization_with_coalescing(gv);
    CHECK(
        std::get<Serial>(result) ==
        std::get<Serial>(expected)); // currently cannot directly compare the 2.
  }

  TEST_CASE("Dependency Invariant SP-ization algorithm - Diamond Pattern") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    std::vector<Node> n = add_nodes(g, 5);
    g.add_edge({n[0], n[1]});
    g.add_edge({n[0], n[2]});
    g.add_edge({n[1], n[3]});
    g.add_edge({n[2], n[3]});
    g.add_edge({n[2], n[4]});
    g.add_edge({n[3], n[4]});
    auto gv = flipped(g); // flipped to account for the diedge bug
    SUBCASE("Naive Version") {
        Serial result =
            std::get<Serial>(dependency_invariant_sp_ization(gv));
        Serial sp0 = {{n[0]}};
        Serial sp1 = {{n[0], n[1]}};
        Serial sp2 = {{n[0], n[2]}};
        Serial sp3 = {{Parallel{{sp2, sp1}}, n[3]}};
        Serial expected = {{Parallel{{sp3, sp2}}, n[4]}};
        CHECK(result == expected);
    }
    SUBCASE("Node coalescing") {
      Node s0 = n[0];
      Parallel p = {{ Serial{{ Parallel{{n[2],n[1]}}, n[3]}},  n[2]}};
      Node s1 = n[4];

      Serial expected = {{s0, p, s1}};

      Serial result = std::get<Serial>(dependency_invariant_sp_ization_with_coalescing(gv));
      CHECK(result == expected);
    }
  }

  TEST_CASE("Dependency Invariant SP-ization algorithm - NASNET-A like") {
    // From the TASO paper, pg 57
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    auto root = add_nodes(g, 1)[0];
    auto input = add_nodes(g, 2);
    auto dwc = add_nodes(g, 5);
    auto conv = add_nodes(g, 5);
    auto avg = add_nodes(g, 3);
    auto add = add_nodes(g, 5);
    auto concat = add_nodes(g, 1)[0];

    g.add_edge({root, input[0]});
    g.add_edge({root, input[1]});

    g.add_edge({input[0], dwc[0]});
    g.add_edge({input[0], dwc[1]});
    g.add_edge({input[0], avg[0]});
    g.add_edge({input[0], avg[1]});
    g.add_edge({input[0], avg[2]});
    g.add_edge({input[0], dwc[2]});
    g.add_edge({input[1], add[2]});
    g.add_edge({input[1], dwc[3]});
    g.add_edge({input[1], dwc[4]});
    g.add_edge({input[1], add[4]});

    g.add_edge({dwc[0], conv[0]});
    g.add_edge({dwc[1], conv[1]});
    g.add_edge({dwc[2], conv[2]});
    g.add_edge({dwc[3], conv[3]});
    g.add_edge({dwc[4], conv[4]});

    g.add_edge({conv[0], add[0]});
    g.add_edge({conv[1], add[0]});
    g.add_edge({avg[0], add[1]});
    g.add_edge({avg[1], add[1]});
    g.add_edge({avg[2], add[2]});
    g.add_edge({conv[2], add[3]});
    g.add_edge({conv[3], add[3]});
    g.add_edge({conv[4], add[4]});

    for (auto const &a : add) {
      g.add_edge({a, concat});
    }
    SUBCASE("No coalescing") {
    DiGraphView gv = flipped(g);
    SerialParallelDecomposition result =
        dependency_invariant_sp_ization(gv);

    int extranodes =
        get_nodes(multidigraph_from_sp_decomposition(result)).size() -
        get_nodes(gv).size();
    CHECK(extranodes >= 0);
    }
    SUBCASE("coalescing") {
          DiGraphView gv = flipped(g);
    SerialParallelDecomposition result =
        dependency_invariant_sp_ization_with_coalescing(gv);

    int extranodes =
        get_nodes(multidigraph_from_sp_decomposition(result)).size() -
        get_nodes(g).size();
    CHECK(extranodes >= 0);
    } 
  }
}
