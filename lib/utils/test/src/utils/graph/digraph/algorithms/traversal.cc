#include "utils/graph/traversal.h"
#include "utils/fmt/vector.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/digraph.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/hash/vector.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_unchecked_dfs_ordering") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    std::vector<Node> n = add_nodes(g, 4);
    add_edges(g,
              {DirectedEdge{n[0], n[1]},
               DirectedEdge{n[1], n[2]},
               DirectedEdge{n[2], n[3]}});

    SUBCASE("simple path") {
      std::vector<Node> correct = {n[0], n[1], n[2], n[3]};
      std::vector<Node> result = get_unchecked_dfs_ordering(g, {n[0]});
      CHECK(correct == result);
    }
  }

  TEST_CASE("get_bfs_ordering") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    std::vector<Node> n = add_nodes(g, 6);
    add_edges(g,
              {DirectedEdge{n[0], n[1]},
               DirectedEdge{n[0], n[2]},
               DirectedEdge{n[1], n[3]},
               DirectedEdge{n[2], n[3]},
               DirectedEdge{n[3], n[4]},
               DirectedEdge{n[4], n[5]}});

    SUBCASE("branching path") {
      std::unordered_set<std::vector<Node>> corrects = {
          {n[0], n[1], n[2], n[3], n[4], n[5]},
          {n[0], n[2], n[1], n[3], n[4], n[5]}};
      std::vector<Node> result = get_bfs_ordering(g, {n[0]});
      CHECK(contains(corrects, result));
    }

    SUBCASE("isolated node") {
      std::vector<Node> correct = {n[5]};
      std::vector<Node> result = get_bfs_ordering(g, {n[5]});
      CHECK(correct == result);
    }

    SUBCASE("graph with cycle") {
      g = DiGraph::create<AdjacencyDiGraph>();
      n = add_nodes(g, 3);
      add_edges(g,
                {DirectedEdge{n[0], n[1]},
                 DirectedEdge{n[0], n[2]},
                 DirectedEdge{n[1], n[0]},
                 DirectedEdge{n[1], n[2]},
                 DirectedEdge{n[2], n[0]},
                 DirectedEdge{n[2], n[1]}});
      std::unordered_set<std::vector<Node>> corrects = {{n[0], n[1], n[2]},
                                                        {n[0], n[2], n[1]}};
      std::vector<Node> result = get_bfs_ordering(g, {n[0]});
      CHECK(contains(corrects, result));
    }
  }

  TEST_CASE("get_dfs_ordering") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    std::vector<Node> n = add_nodes(g, 4);
    add_edges(g,
              {DirectedEdge{n[0], n[1]},
               DirectedEdge{n[1], n[2]},
               DirectedEdge{n[2], n[3]}});

    SUBCASE("simple path") {
      std::vector<Node> correct = {n[0], n[1], n[2], n[3]};
      std::vector<Node> result = get_dfs_ordering(g, {n[0]});
      CHECK(correct == result);
    }

    SUBCASE("with cycle") {
      g.add_edge(DirectedEdge{n[3], n[1]});
      std::vector<Node> correct = {n[0], n[1], n[2], n[3]};
      std::vector<Node> result = get_dfs_ordering(g, {n[0]});
      CHECK(correct == result);
    }

    SUBCASE("branching") {
      g.add_edge(DirectedEdge{n[1], n[3]});
      std::unordered_set<std::vector<Node>> corrects = {
          {n[0], n[1], n[2], n[3]}, {n[0], n[1], n[3], n[2]}};
      std::vector<Node> result = get_dfs_ordering(g, {n[0]});
      CHECK(contains(corrects, result));
    }

    SUBCASE("disconnected") {
      g.remove_edge(DirectedEdge{n[2], n[3]});
      std::vector<Node> correct = {n[0], n[1], n[2]};
      std::vector<Node> result = get_dfs_ordering(g, {n[0]});
      CHECK(correct == result);
    }

    SUBCASE("isolated node") {
      g.remove_edge(DirectedEdge{n[2], n[3]});
      std::vector<Node> correct = {n[3]};
      std::vector<Node> result = get_dfs_ordering(g, {n[3]});
      CHECK(correct == result);
    }
  }
}
