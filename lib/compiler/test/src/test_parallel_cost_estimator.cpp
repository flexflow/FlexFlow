#include "compiler/machine_mapping.h"
#include "doctest/doctest.h"
#include "rapidcheck.h"

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("parallel_estimate_cost") {
    // Test graph structure
    //     /-- 1 --\
        //  0 -         - 3
    //     \-- 2 --/
    OutputLabelledOpenMultiDiGraph<int, int> g =
        OutputLabelledOpenMultiDiGraph<int, int>::create<
            UnorderedOutputLabelledOpenMultiDiGraph<int, int>>();
    Node n0 = g.add_node(0);
    Node n1 = g.add_node(1);
    Node n2 = g.add_node(2);
    Node n3 = g.add_node(3);

    NodePort p0 = g.add_node_port();
    NodePort p1 = g.add_node_port();
    NodePort p2 = g.add_node_port();
    NodePort p3 = g.add_node_port();
    NodePort p4 = g.add_node_port();
    NodePort p5 = g.add_node_port();
    NodePort p6 = g.add_node_port();
    NodePort p7 = g.add_node_port();

    // MultiDiEdge: dst, dstport, src, srcport
    MultiDiEdge e0{n1, p1, n0, p0};
    MultiDiEdge e1{n2, p2, n0, p0};
    MultiDiEdge e2{n3, p5, n1, p3};
    MultiDiEdge e3{n3, p6, n2, p4};

    g.add_edge(e0);
    g.add_edge(e1);
    g.add_edge(e2);
    g.add_edge(e3);
    g.add_edge(e4);

    g.add_label(e0, 10);
    g.add_label(e1, 11);
    g.add_label(e2, 12);
    g.add_label(e3, 13);
    g.add_label(e4, 14);

    std::unordered_set node_set{n1, n4};
    auto subgraph = get_subgraph()
  }
}
