#include "utils/graph/series_parallel/parallel_reduction.h"
#include "utils/graph/instances/adjacency_multidigraph.h"
#include "utils/graph/multidigraph/algorithms/add_edges.h"
#include "utils/graph/multidigraph/algorithms/add_nodes.h"
#include "utils/graph/multidigraph/algorithms/get_directed_edge.h"
#include "utils/graph/multidigraph/algorithms/get_edge_counts.h"
#include "utils/graph/multidigraph/algorithms/get_edges.h"
#include "utils/graph/node/algorithms.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("find_parallel_reduction") {
    MultiDiGraph g = MultiDiGraph::create<AdjacencyMultiDiGraph>();
    SUBCASE("base case") {
      std::vector<Node> n = add_nodes(g, 2);
      std::vector<MultiDiEdge> e = add_edges(g,
                                             {
                                                 {n.at(0), n.at(1)},
                                                 {n.at(0), n.at(1)},
                                             });

      std::optional<ParallelReduction> result = find_parallel_reduction(g);
      std::optional<ParallelReduction> correct =
          make_parallel_reduction(e.at(0), e.at(1));
      CHECK(result == correct);
    }

    SUBCASE("does not apply when there is only one edge") {
      std::vector<Node> n = add_nodes(g, 2);
      std::vector<MultiDiEdge> e = add_edges(g,
                                             {
                                                 {n.at(0), n.at(1)},
                                             });

      std::optional<ParallelReduction> result = find_parallel_reduction(g);
      std::optional<ParallelReduction> correct = std::nullopt;
      CHECK(result == correct);
    }

    SUBCASE("requires both ends be the same") {
      std::vector<Node> n = add_nodes(g, 3);
      SUBCASE("branch out") {
        std::vector<MultiDiEdge> e = add_edges(g,
                                               {
                                                   {n.at(0), n.at(1)},
                                                   {n.at(0), n.at(2)},
                                               });

        std::optional<ParallelReduction> result = find_parallel_reduction(g);
        std::optional<ParallelReduction> correct = std::nullopt;
        CHECK(result == correct);
      }

      SUBCASE("branch in") {
        std::vector<MultiDiEdge> e = add_edges(g,
                                               {
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
      std::vector<MultiDiEdge> e = add_edges(g,
                                             {
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
      std::vector<MultiDiEdge> e = add_edges(g,
                                             {
                                                 {n.at(0), n.at(1)},
                                                 {n.at(0), n.at(2)},
                                                 {n.at(0), n.at(3)},
                                                 {n.at(1), n.at(3)},
                                                 {n.at(1), n.at(3)},
                                                 {n.at(2), n.at(4)},
                                                 {n.at(3), n.at(4)},
                                             });

      std::optional<ParallelReduction> result = find_parallel_reduction(g);
      std::optional<ParallelReduction> correct =
          make_parallel_reduction(e.at(3), e.at(4));
      CHECK(result == correct);
    }
  }

  TEST_CASE("apply_parallel_reduction") {
    MultiDiGraph g = MultiDiGraph::create<AdjacencyMultiDiGraph>();

    SUBCASE("base case") {
      std::vector<Node> n = add_nodes(g, 2);
      std::vector<MultiDiEdge> e = add_edges(g,
                                             {
                                                 {n.at(0), n.at(1)},
                                                 {n.at(0), n.at(1)},
                                             });

      ParallelReduction input = make_parallel_reduction(e.at(0), e.at(1));

      MultiDiEdge returned_edge = apply_parallel_reduction(g, input);

      SUBCASE("nodes") {
        std::unordered_set<Node> result_nodes = get_nodes(g);
        std::unordered_set<Node> correct_nodes = unordered_set_of(n);
        CHECK(result_nodes == correct_nodes);
      }

      SUBCASE("edge shape") {
        std::unordered_map<DirectedEdge, int> result_edges = get_edge_counts(g);
        std::unordered_map<DirectedEdge, int> correct_edges = {
            {DirectedEdge{n.at(0), n.at(1)}, 1},
        };
        CHECK(result_edges == correct_edges);
      }

      SUBCASE("return value and edge ids") {
        std::unordered_set<MultiDiEdge> result_edge_ids = get_edges(g);
        std::unordered_set<MultiDiEdge> correct_edge_ids = {returned_edge};
        CHECK(result_edge_ids == correct_edge_ids);
      }
    }

    SUBCASE("in larger graph") {
      std::vector<Node> n = add_nodes(g, 5);
      std::vector<MultiDiEdge> e = add_edges(g,
                                             {
                                                 {n.at(0), n.at(1)},
                                                 {n.at(0), n.at(2)},
                                                 {n.at(0), n.at(3)},
                                                 {n.at(1), n.at(3)},
                                                 {n.at(1), n.at(3)},
                                                 {n.at(2), n.at(4)},
                                                 {n.at(3), n.at(4)},
                                             });

      std::unordered_map<DirectedEdge, int> input_edge_counts =
          get_edge_counts(g);

      MultiDiEdge reduction_e1 = e.at(3);
      MultiDiEdge reduction_e2 = e.at(4);
      ParallelReduction input =
          make_parallel_reduction(reduction_e1, reduction_e2);

      MultiDiEdge returned_edge = apply_parallel_reduction(g, input);

      SUBCASE("nodes") {
        std::unordered_set<Node> result_nodes = get_nodes(g);
        std::unordered_set<Node> correct_nodes = unordered_set_of(n);
        CHECK(result_nodes == correct_nodes);
      }

      SUBCASE("edge shape") {
        std::unordered_map<DirectedEdge, int> result_edges = get_edge_counts(g);
        std::unordered_map<DirectedEdge, int> correct_edges = [&] {
          std::unordered_map<DirectedEdge, int> new_edge_counts =
              input_edge_counts;
          new_edge_counts.at(get_directed_edge(g, reduction_e1))--;
          return new_edge_counts;
        }();
        CHECK(result_edges == correct_edges);
      }

      SUBCASE("return value and edge ids") {
        std::unordered_set<MultiDiEdge> result_edge_ids = get_edges(g);
        std::unordered_set<MultiDiEdge> correct_edge_ids = [&] {
          std::unordered_set<MultiDiEdge> new_edges = unordered_set_of(e);
          new_edges.erase(reduction_e1);
          new_edges.erase(reduction_e2);
          new_edges.insert(returned_edge);
          return new_edges;
        }();
        CHECK(result_edge_ids == correct_edge_ids);
      }
    }
  }
}
