#include "utils/graph/serial_parallel/sp_ization/critical_path_preserving_sp_ization.h"
#include "test/utils/doctest.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/digraph/digraph.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"
#include "utils/graph/serial_parallel/serial_parallel_metrics.h"
#include "utils/graph/serial_parallel/serial_parallel_splits.h"

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("critical_path_preserving_sp_ization") {

    SUBCASE("Sample Graph #1") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 6);
      add_edges(g,
                {
                    DirectedEdge{n[0], n[1]},
                    DirectedEdge{n[0], n[2]},
                    DirectedEdge{n[1], n[2]},
                    DirectedEdge{n[1], n[3]},
                    DirectedEdge{n[2], n[4]},
                    DirectedEdge{n[3], n[4]},
                    DirectedEdge{n[3], n[5]},
                    DirectedEdge{n[4], n[5]},
                });

      std::unordered_map<Node, float> cost_map = {
          {n[0], 3}, {n[1], 2}, {n[2], 1}, {n[3], 1}, {n[4], 1}, {n[5], 5}};

      CHECK(work_cost(g, cost_map) == 13);
      CHECK(critical_path_cost(g, cost_map) == 12);

      SerialParallelDecomposition sp = critical_path_preserving_sp_ization(g);

      SUBCASE("structure") {
        Node sp0 = n[0];
        SerialSplit sp1 = SerialSplit{sp0, n[1]};
        SerialSplit sp2 = SerialSplit{ParallelSplit{sp0, sp1}, n[2]};
        SerialSplit sp3 = SerialSplit{n[0], n[1], n[3]};
        SerialSplit sp4 = SerialSplit{ParallelSplit{sp2, sp3}, n[4]};
        SerialSplit sp5 = SerialSplit{ParallelSplit{sp3, sp4}, n[5]};
        SerialParallelDecomposition expected = sp5;
        SerialParallelDecomposition result = sp;
        CHECK(expected == result);
      }
      SUBCASE("work cost") {
        float expected = 3 * 4 + 2 * 3 + 1 * 1 + 1 * 2 + 1 * 1 + 5 * 1;
        float result = work_cost(sp, cost_map);
        CHECK(expected == result);
      }

      SUBCASE("critical path cost") {
        float expected = critical_path_cost(g, cost_map);
        float result = critical_path_cost(sp, cost_map);
        CHECK(expected == result);
      }
    }

    SUBCASE("Sample Graph #2") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 6);
      add_edges(g,
                {
                    DirectedEdge{n[0], n[1]},
                    DirectedEdge{n[0], n[2]},
                    DirectedEdge{n[1], n[3]},
                    DirectedEdge{n[2], n[5]},
                    DirectedEdge{n[3], n[4]},
                    DirectedEdge{n[4], n[5]},
                });

      std::unordered_map<Node, float> cost_map = {
          {n[0], 1}, {n[1], 1}, {n[2], 10}, {n[3], 1}, {n[4], 1}, {n[5], 1}};

      CHECK(work_cost(g, cost_map) == 15);
      CHECK(critical_path_cost(g, cost_map) == 12);

      SerialParallelDecomposition sp = critical_path_preserving_sp_ization(g);

      SUBCASE("structure") {
        SerialParallelDecomposition expected =
            SerialSplit{ParallelSplit{SerialSplit{n[0], n[1], n[3], n[4]},
                                      SerialSplit{n[0], n[2]}},
                        n[5]};
        SerialParallelDecomposition result = sp;
        CHECK(expected == result);
      }
      SUBCASE("work cost") {
        float expected = 16;
        float result = work_cost(sp, cost_map);
        CHECK(expected == result);
      }

      SUBCASE("critical path cost") {
        float expected = critical_path_cost(g, cost_map);
        float result = critical_path_cost(sp, cost_map);
        CHECK(expected == result);
      }
    }
  }

  TEST_CASE("critical_path_preserving_sp_ization_with_coalescing") {

    SUBCASE("Sample Graph #1") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 6);
      add_edges(g,
                {
                    DirectedEdge{n[0], n[1]},
                    DirectedEdge{n[0], n[2]},
                    DirectedEdge{n[1], n[2]},
                    DirectedEdge{n[1], n[3]},
                    DirectedEdge{n[2], n[4]},
                    DirectedEdge{n[3], n[4]},
                    DirectedEdge{n[3], n[5]},
                    DirectedEdge{n[4], n[5]},
                });

      std::unordered_map<Node, float> cost_map = {
          {n[0], 1}, {n[1], 1}, {n[2], 2}, {n[3], 3}, {n[4], 1}, {n[5], 1}};

      CHECK(work_cost(g, cost_map) == 9);
      CHECK(critical_path_cost(g, cost_map) == 7);

      SerialParallelDecomposition sp =
          critical_path_preserving_sp_ization_with_coalescing(g);

      SUBCASE("structure") {
        SerialParallelDecomposition expected = SerialSplit{
            n[0],
            n[1],
            ParallelSplit{SerialSplit{ParallelSplit{n[2], n[3]}, n[4]}, n[3]},
            n[5]};
        SerialParallelDecomposition result = sp;
        CHECK(expected == result);
      }
      SUBCASE("work cost") {
        float expected = 12;
        float result = work_cost(sp, cost_map);
        CHECK(expected == result);
      }

      SUBCASE("critical path cost") {
        float expected = critical_path_cost(g, cost_map);
        float result = critical_path_cost(sp, cost_map);
        CHECK(expected == result);
      }
    }

    SUBCASE("Sample Graph #2") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 6);
      add_edges(g,
                {
                    DirectedEdge{n[0], n[1]},
                    DirectedEdge{n[0], n[2]},
                    DirectedEdge{n[1], n[3]},
                    DirectedEdge{n[2], n[5]},
                    DirectedEdge{n[3], n[4]},
                    DirectedEdge{n[4], n[5]},
                });

      std::unordered_map<Node, float> cost_map = {
          {n[0], 1}, {n[1], 1}, {n[2], 10}, {n[3], 1}, {n[4], 1}, {n[5], 1}};

      CHECK(work_cost(g, cost_map) == 15);
      CHECK(critical_path_cost(g, cost_map) == 12);

      SerialParallelDecomposition sp =
          critical_path_preserving_sp_ization_with_coalescing(g);

      SUBCASE("structure") {
        SerialParallelDecomposition expected = SerialSplit{
            n[0], ParallelSplit{SerialSplit{n[1], n[3], n[4]}, n[2]}, n[5]};
        SerialParallelDecomposition result = sp;
        CHECK(expected == result);
      }
      SUBCASE("work cost") {
        float expected = 15;
        float result = work_cost(sp, cost_map);
        CHECK(expected == result);
      }

      SUBCASE("critical path cost") {
        float expected = critical_path_cost(g, cost_map);
        float result = critical_path_cost(sp, cost_map);
        CHECK(expected == result);
      }
    }

    SUBCASE("Sample Graph #3") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 10);
      add_edges(g,
                {DirectedEdge{n[0], n[1]},
                 DirectedEdge{n[0], n[3]},
                 DirectedEdge{n[1], n[2]},
                 DirectedEdge{n[1], n[5]},
                 DirectedEdge{n[1], n[4]},
                 DirectedEdge{n[2], n[6]},
                 DirectedEdge{n[3], n[4]},
                 DirectedEdge{n[3], n[5]},
                 DirectedEdge{n[3], n[8]},
                 DirectedEdge{n[4], n[8]},
                 DirectedEdge{n[5], n[7]},
                 DirectedEdge{n[7], n[8]},
                 DirectedEdge{n[6], n[9]},
                 DirectedEdge{n[8], n[9]}});

      std::unordered_map<Node, float> cost_map = {{n[0], 1},
                                                  {n[1], 1},
                                                  {n[2], 4},
                                                  {n[3], 10},
                                                  {n[4], 10},
                                                  {n[5], 5},
                                                  {n[6], 4},
                                                  {n[7], 3},
                                                  {n[8], 4},
                                                  {n[9], 1}};

      CHECK(work_cost(g, cost_map) == 43);
      CHECK(critical_path_cost(g, cost_map) == 26);

      SerialParallelDecomposition sp =
          critical_path_preserving_sp_ization_with_coalescing(g);

      SUBCASE("structure") {

        SerialParallelDecomposition expected = SerialSplit{
            n[0],
            ParallelSplit{
                SerialSplit{n[1], n[2], n[6]},
                SerialSplit{ParallelSplit{
                                SerialSplit{ParallelSplit{n[1], n[3]},
                                            ParallelSplit{
                                                n[4], SerialSplit{n[5], n[7]}}},
                                n[3]},
                            n[8]}},
            n[9]};
        SerialParallelDecomposition result = sp;
        CHECK(expected == result);
      };
      SUBCASE("work cost") {
        float expected = 54;
        float result = work_cost(sp, cost_map);
        CHECK(expected == result);
      }

      SUBCASE("critical path cost") {
        float expected = critical_path_cost(g, cost_map);
        float result = critical_path_cost(sp, cost_map);
        CHECK(expected == result);
      }
    }
  }
}
