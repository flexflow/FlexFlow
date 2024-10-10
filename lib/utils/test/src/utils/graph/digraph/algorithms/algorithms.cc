#include "utils/graph/digraph/algorithms.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/digraph.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("DiGraph - algorithms.cc") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();

    std::vector<Node> n = add_nodes(g, 4);
    std::vector<DirectedEdge> e = {
        DirectedEdge{n[0], n[1]},
        DirectedEdge{n[0], n[2]},
        DirectedEdge{n[0], n[3]},
        DirectedEdge{n[1], n[2]},
    };
    add_edges(g, e);

    SUBCASE("get_edges") {
      SUBCASE("Base") {
        std::unordered_set<DirectedEdge> correct = unordered_set_of(e);
        std::unordered_set<DirectedEdge> result = get_edges(g);
        CHECK(result == correct);
      }

      SUBCASE("Adding an edge") {
        g.add_edge(DirectedEdge{n[3], n[1]});
        std::unordered_set<DirectedEdge> correct = {
            DirectedEdge{n[0], n[1]},
            DirectedEdge{n[0], n[2]},
            DirectedEdge{n[0], n[3]},
            DirectedEdge{n[1], n[2]},
            DirectedEdge{n[3], n[1]},
        };
        std::unordered_set<DirectedEdge> result = get_edges(g);
        CHECK(result == correct);
      }

      SUBCASE("Removing an edge") {
        g.remove_edge(DirectedEdge{n[0], n[3]});
        std::unordered_set<DirectedEdge> correct = {
            DirectedEdge{n[0], n[1]},
            DirectedEdge{n[0], n[2]},
            DirectedEdge{n[1], n[2]},
        };
        std::unordered_set<DirectedEdge> result = get_edges(g);
        CHECK(result == correct);
      }
    }

    SUBCASE("get_sinks") {
      SUBCASE("Base") {
        std::unordered_set<Node> correct = {n[2], n[3]};
        std::unordered_set<Node> result = get_sinks(g);
        CHECK(result == correct);
      }

      SUBCASE("Adding an edge to remove a sink") {
        g.add_edge(DirectedEdge{n[3], n[2]});
        std::unordered_set<Node> correct = {n[2]};
        std::unordered_set<Node> result = get_sinks(g);
        CHECK(result == correct);
      }

      SUBCASE("Creating a cycle") {
        g.add_edge(DirectedEdge{n[2], n[0]});
        std::unordered_set<Node> result = get_sinks(g);
        std::unordered_set<Node> correct = {n[3]};
        CHECK(result == correct);
      }
    }

    SUBCASE("get_sources") {
      SUBCASE("Base") {
        std::unordered_set<Node> correct = {n[0]};
        std::unordered_set<Node> result = get_sources(g);
        CHECK(result == correct);
      }

      SUBCASE("Adding an edge to remove a source") {
        g.add_edge(DirectedEdge{n[2], n[0]});
        std::unordered_set<Node> correct = {};
        std::unordered_set<Node> result = get_sources(g);
        CHECK(result == correct);
      }

      SUBCASE("Removing an edge to create a new source") {
        g.remove_edge(DirectedEdge{n[0], n[1]});
        std::unordered_set<Node> correct = {n[0], n[1]};
        std::unordered_set<Node> result = get_sources(g);
        CHECK(result == correct);
      }

      SUBCASE("Creating a cycle") {
        g.add_edge(DirectedEdge{n[2], n[0]});
        std::unordered_set<Node> result = get_sources(g);
        std::unordered_set<Node> correct = {};
        CHECK(result.empty());
      }
    }
  }
}
