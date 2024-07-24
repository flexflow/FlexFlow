#include <doctest/doctest.h>
#include "utils/graph/serial_parallel/series_reduction.h"
#include "utils/graph/instances/adjacency_multidigraph.h"
#include "utils/graph/multidigraph/algorithms/add_nodes.h"
#include "utils/graph/multidigraph/algorithms/add_edges.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("find_series_reduction") {
    MultiDiGraph g = MultiDiGraph::create<AdjacencyMultiDiGraph>();
    SUBCASE("base case") {
      std::vector<Node> n = add_nodes(g, 3);  
      std::vector<MultiDiEdge> e = add_edges(g, {
        {n.at(0), n.at(1)},
        {n.at(1), n.at(2)},
      });

      std::optional<SeriesReduction> result = find_series_reduction(g);
      std::optional<SeriesReduction> correct = make_series_reduction(e.at(0), e.at(1));
      CHECK(result == correct);
    }

    SUBCASE("does not find if other edges are involved with center node") {
      SUBCASE("duplicate edge") {
        std::vector<Node> n = add_nodes(g, 3);
        std::vector<MultiDiEdge> e = add_edges(g, {
          {n.at(0), n.at(1)},
          {n.at(0), n.at(1)},
          {n.at(1), n.at(2)},
        });

        std::optional<SeriesReduction> result = find_series_reduction(g);
        std::optional<SeriesReduction> correct = std::nullopt;
        CHECK(result == correct);
      }

      SUBCASE("misc edge") {
        std::vector<Node> n = add_nodes(g, 4);
        std::vector<MultiDiEdge> e = add_edges(g, {
          {n.at(0), n.at(1)},
          {n.at(1), n.at(3)},
          {n.at(1), n.at(2)},
        });

        std::optional<SeriesReduction> result = find_series_reduction(g);
        std::optional<SeriesReduction> correct = std::nullopt;
        CHECK(result == correct);
      }
    }

    SUBCASE("does find if other edges are involved with non-center node") {
      std::vector<Node> n = add_nodes(g, 4);
      SUBCASE("edge from dst") {
        std::vector<MultiDiEdge> e = add_edges(g, {
          {n.at(0), n.at(1)},
          {n.at(1), n.at(2)},

          {n.at(2), n.at(3)},
          {n.at(3), n.at(2)},
          {n.at(0), n.at(3)},
          {n.at(3), n.at(0)},
        });

        std::optional<SeriesReduction> result = find_series_reduction(g);
        std::optional<SeriesReduction> correct = make_series_reduction(e.at(0), e.at(1));
        CHECK(result == correct);
      }
    }

    SUBCASE("finds one reduction when there are multiple") {
      std::vector<Node> n = add_nodes(g, 4);
      std::vector<MultiDiEdge> e = add_edges(g, {
        {n.at(0), n.at(1)},
        {n.at(1), n.at(2)},
        {n.at(2), n.at(3)},
      });

      std::optional<SeriesReduction> result = find_series_reduction(g);
      std::unordered_set<SeriesReduction> correct_options = {
        make_series_reduction(e.at(0), e.at(1)),
        make_series_reduction(e.at(1), e.at(2)),
      };
      CHECK(result.has_value());
      CHECK(contains(correct_options, result.value()));
    }

    SUBCASE("in larger graph") {
      std::vector<Node> n = add_nodes(g, 8);
      std::vector<MultiDiEdge> e = add_edges(g, {
        {n.at(0), n.at(2)},
        {n.at(1), n.at(2)},
        {n.at(2), n.at(3)},
        {n.at(2), n.at(3)},
        {n.at(3), n.at(4)}, //*
        {n.at(4), n.at(5)}, //*
        {n.at(5), n.at(6)},
        {n.at(5), n.at(7)},
      });

      std::optional<SeriesReduction> result = find_series_reduction(g);
      std::optional<SeriesReduction> correct = make_series_reduction(e.at(4), e.at(5));
      CHECK(result == correct);
    }
  }
}
