#include <doctest/doctest.h>
#include "utils/graph/serial_parallel/parallel_reduction.h"
#include "utils/graph/instances/adjacency_multidigraph.h"
#include "utils/graph/multidigraph/algorithms/add_nodes.h"
#include "utils/graph/multidigraph/algorithms/add_edges.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("find_parallel_reduction") {
    MultiDiGraph g = MultiDiGraph::create<AdjacencyMultiDiGraph>();
    SUBCASE("base case") {
      std::vector<Node> n = add_nodes(g, 2);
      std::vector<MultiDiEdge> e = add_edges(g, {
        {n.at(0), n.at(1)},
        {n.at(0), n.at(1)},
      });

      std::optional<ParallelReduction> result = find_parallel_reduction(g);
      std::optional<ParallelReduction> correct = make_parallel_reduction(e.at(0), e.at(1));
      CHECK(result == correct);
    }

    SUBCASE("does not apply when there is only one edge") {
      std::vector<Node> n = add_nodes(g, 2);
      std::vector<MultiDiEdge> e = add_edges(g, {
        {n.at(0), n.at(1)},
      });

      std::optional<ParallelReduction> result = find_parallel_reduction(g);
      std::optional<ParallelReduction> correct = std::nullopt;
      CHECK(result == correct);
    }

    SUBCASE("requires both ends be the same") {
      std::vector<Node> n = add_nodes(g, 3);
        SUBCASE("branch out") {
        std::vector<MultiDiEdge> e = add_edges(g, {
          {n.at(0), n.at(1)},
          {n.at(0), n.at(2)},
        });

        std::optional<ParallelReduction> result = find_parallel_reduction(g);
        std::optional<ParallelReduction> correct = std::nullopt;
        CHECK(result == correct);
      }

      SUBCASE("branch in") {
        std::vector<MultiDiEdge> e = add_edges(g, {
          {n.at(0), n.at(2)},
          {n.at(1), n.at(2)},
        });

        std::optional<ParallelReduction> result = find_parallel_reduction(g);
        std::optional<ParallelReduction> correct = std::nullopt;
        CHECK(result == correct);
      }
    }

    SUBCASE("finds one reduction when there are multiple") {
      std::vector<Node> n = add_nodes(g, 2);
      std::vector<MultiDiEdge> e = add_edges(g, {
        {n.at(0), n.at(1)},
        {n.at(0), n.at(1)},
        {n.at(0), n.at(1)},
      });

      std::optional<ParallelReduction> result = find_parallel_reduction(g);
      std::unordered_set<ParallelReduction> correct_options = {
        make_parallel_reduction(e.at(0), e.at(1)),
        make_parallel_reduction(e.at(1), e.at(2)),
        make_parallel_reduction(e.at(0), e.at(2)),
      };
      CHECK(result.has_value());
      CHECK(contains(correct_options, result.value()));
    }

    SUBCASE("in larger graph") {
      std::vector<Node> n = add_nodes(g, 5);
      std::vector<MultiDiEdge> e = add_edges(g, {
        {n.at(0), n.at(1)},
        {n.at(0), n.at(2)},
        {n.at(0), n.at(3)},
        {n.at(1), n.at(3)},
        {n.at(1), n.at(3)},
        {n.at(2), n.at(4)},
        {n.at(3), n.at(4)},
      });
      
      std::optional<ParallelReduction> result = find_parallel_reduction(g);
      std::optional<ParallelReduction> correct = make_parallel_reduction(e.at(3), e.at(4));
      CHECK(result == correct);
    }
  }
}
