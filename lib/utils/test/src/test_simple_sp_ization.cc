#include "test/utils/doctest.h"
#include "utils/graph/adjacency_digraph.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph.h"
#include "utils/graph/simple_sp_ization.h"

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("Simple SP-ization: sanity checks") {
    /*
    graph TD;
    N1[N1] --> N2[N2];
    N1[N1] --> N3[N3];
    N2[N2] --> N4[N4];
    N2[N2] --> N5[N5];
    N3[N3] --> N4[N4];
    N3[N3] --> N5[N5];
    N4[N4] --> N6[N6];
    N5[N5] --> N6[N6];
    */
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    std::vector<Node> nodes = add_nodes(g, 6);

    g.add_edge({nodes[0], nodes[1]});
    g.add_edge({nodes[0], nodes[2]});

    g.add_edge({nodes[1], nodes[3]});
    g.add_edge({nodes[1], nodes[4]});
    g.add_edge({nodes[2], nodes[3]});
    g.add_edge({nodes[2], nodes[4]});

    g.add_edge({nodes[3], nodes[5]});
    g.add_edge({nodes[4], nodes[5]});

    DiGraph sp = simple_sp_ization(g);

    CHECK(is_acyclic(sp));

    CHECK(get_sources(sp).size() == 1);

    CHECK(get_sinks(sp).size() == 1);

    std::unordered_set<Node> sp_nodes = get_nodes(sp);
    CHECK(sp_nodes.size() ==
          6 + 3); // Original nodes + 3 barrier nodes (one per layer transition)

    std::unordered_set<DirectedEdge> sp_edges = get_edges(sp);
    CHECK(sp_edges.size() == 3 + 4 + 3); // 3 for first layer, 4 for second, ...
  }

  TEST_CASE("Simple SP-ization: integrity checks") {
    /*
    graph TD;
    N1[N1] --> N2[N2];
    N1[N1] --> N3[N3];
    N2[N2] --> N4[N4];
    N2[N2] --> N5[N5];
    N3[N3] --> N4[N4];
    N3[N3] --> N5[N5];
    N4[N4] --> N6[N6];
    N5[N5] --> N6[N6];
    */
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    std::vector<Node> nodes = add_nodes(g, 6);

    g.add_edge({nodes[0], nodes[1]});
    g.add_edge({nodes[0], nodes[2]});

    g.add_edge({nodes[1], nodes[3]});
    g.add_edge({nodes[1], nodes[4]});
    g.add_edge({nodes[2], nodes[3]});
    g.add_edge({nodes[2], nodes[4]});

    g.add_edge({nodes[3], nodes[5]});
    g.add_edge({nodes[4], nodes[5]});

    DiGraph sp = simple_sp_ization(g);

    Node source = *(get_sources(sp).begin());

    auto s1 = get_successors(sp, source);
    CHECK(s1.size() == 1);
    Node b1 = *(s1.begin()); // Barrier Node

    auto layer2 = get_successors(sp, b1);
    CHECK(layer2.size() == 2);
    Node b2 = *(get_successors(sp, *(layer2.begin())).begin());
    CHECK(get_predecessors(sp, b2).size() == 2);

    auto layer3 = get_successors(sp, b2);
    CHECK(layer3.size() == 2);
    Node b3 = *(get_successors(sp, *(layer3.begin())).begin());
    CHECK(get_predecessors(sp, b3).size() == 2);

    auto layer4 = get_successors(sp, b3);
    CHECK(layer4.size() == 1);
    CHECK(get_successors(sp, *(layer4.begin())).size() == 0); // Last node
  }
}
