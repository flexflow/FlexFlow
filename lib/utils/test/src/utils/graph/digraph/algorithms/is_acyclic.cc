#include "utils/graph/digraph/algorithms/is_acyclic.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
    TEST_CASE("is_acyclic - empty graph") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
        CHECK(is_acyclic(g));
    }

    TEST_CASE("is_acyclic - single node") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
        add_nodes(g, 1);
        CHECK(is_acyclic(g));
    }

    TEST_CASE("is_acyclic - simple acyclic graph") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
        std::vector<Node> n = add_nodes(g, 3);
        add_edges(g, {
            DirectedEdge{n[0], n[1]},
            DirectedEdge{n[1], n[2]},
        });
        CHECK(is_acyclic(g));
    }

    TEST_CASE("is_acyclic - simple cyclic graph") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
        std::vector<Node> n = add_nodes(g, 3);
        add_edges(g, {
            DirectedEdge{n[0], n[1]},
            DirectedEdge{n[1], n[2]},
            DirectedEdge{n[2], n[0]},
        });
        CHECK_FALSE(is_acyclic(g));
    }

    TEST_CASE("is_acyclic - 2 parallel chains") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
        std::vector<Node> n = add_nodes(g, 6);
        add_edges(g, {
            DirectedEdge{n[0], n[1]},
            DirectedEdge{n[0], n[2]},
            DirectedEdge{n[1], n[3]},
            DirectedEdge{n[2], n[4]},
            DirectedEdge{n[3], n[5]},
            DirectedEdge{n[4], n[5]},
        });
        CHECK(is_acyclic(g));
    }

    TEST_CASE("is_acyclic - complex cyclic graph") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
        std::vector<Node> n = add_nodes(g, 6);
        add_edges(g, {
            DirectedEdge{n[0], n[1]},
            DirectedEdge{n[0], n[2]},
            DirectedEdge{n[1], n[3]},
            DirectedEdge{n[2], n[4]},
            DirectedEdge{n[3], n[5]},
            DirectedEdge{n[4], n[5]},
            DirectedEdge{n[5], n[1]},
        });
        CHECK_FALSE(is_acyclic(g));
    }

    TEST_CASE("is_acyclic - complex cyclic graph 2") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
        std::vector<Node> n = add_nodes(g, 6);
        add_edges(g, {
            DirectedEdge{n[0], n[1]},
            DirectedEdge{n[1], n[2]},
            DirectedEdge{n[1], n[3]},
            DirectedEdge{n[1], n[5]},
            DirectedEdge{n[2], n[4]},
            DirectedEdge{n[3], n[1]},
            DirectedEdge{n[3], n[4]},
        });
        CHECK_FALSE(is_acyclic(g));
    }

    TEST_CASE("traversal") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
        std::vector<Node> n = add_nodes(g, 5);
        add_edges(g, {
            DirectedEdge{n[0], n[1]},
            DirectedEdge{n[1], n[2]},
            DirectedEdge{n[2], n[3]}
        });

        SUBCASE("with root") {
            g.add_edge(DirectedEdge{n[3], n[2]});
            CHECK_FALSE(is_acyclic(g));
        }

        SUBCASE("without root") {
            g.add_edge(DirectedEdge{n[3], n[0]});
            CHECK_FALSE(is_acyclic(g));
        }

        SUBCASE("nonlinear") {
            g.add_edge(DirectedEdge{n[1], n[3]});
            CHECK(is_acyclic(g));
        }
    }
}