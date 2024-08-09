#include "utils/graph/serial_parallel/sp_ization/work_preserving_sp_ization.h"
#include "test/utils/doctest.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/digraph/digraph.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"
#include "utils/graph/serial_parallel/serial_parallel_metrics.h"
#include "utils/graph/serial_parallel/serial_parallel_splits.h"

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("work_preserving_") {

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

      SerialParallelDecomposition sp = stratum_sync_sp_ization(g);

      SUBCASE("structure") {
        SerialParallelDecomposition correct(
            SerialSplit{n[0], n[1], ParallelSplit{n[2], n[3]}, n[4], n[5]});
        SerialParallelDecomposition result = sp;
        CHECK(correct == result);
      }
      SUBCASE("work cost") {
        float correct = work_cost(g, cost_map);
        float result = work_cost(sp, cost_map);
        CHECK(correct == result);
      }

      SUBCASE("critical path cost") {
        float correct = 7;
        float result = critical_path_cost(sp, cost_map);
        CHECK(correct == result);
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

      SerialParallelDecomposition sp = stratum_sync_sp_ization(g);

      SUBCASE("structure") {
        SerialParallelDecomposition correct(
            SerialSplit{n[0], ParallelSplit{n[1], n[2]}, n[3], n[4], n[5]});
        SerialParallelDecomposition result = sp;
        CHECK(correct == result);
      }
      SUBCASE("work cost") {
        float correct = work_cost(g, cost_map);
        float result = work_cost(sp, cost_map);
        CHECK(correct == result);
      }

      SUBCASE("critical path cost") {
        float correct = 14;
        float result = critical_path_cost(sp, cost_map);
        CHECK(correct == result);
      }
    }

    SUBCASE("Sample Graph #3") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 9);
      add_edges(g,
                {
                    DirectedEdge{n[0], n[1]},
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
                });

      std::unordered_map<Node, float> cost_map = {{n[0], 1},
                                                  {n[1], 1},
                                                  {n[2], 10},
                                                  {n[3], 10},
                                                  {n[4], 1},
                                                  {n[5], 1},
                                                  {n[6], 10},
                                                  {n[7], 10},
                                                  {n[8], 1}};

      CHECK(work_cost(g, cost_map) == 45);
      CHECK(critical_path_cost(g, cost_map) == 23);

      SerialParallelDecomposition sp = stratum_sync_sp_ization(g);

      SUBCASE("structure") {
        SerialParallelDecomposition correct(
            SerialSplit{n[0],
                        ParallelSplit{n[1], n[3]},
                        ParallelSplit{n[2], n[4], n[5]},
                        ParallelSplit{n[6], n[7]},
                        n[8]});
        SerialParallelDecomposition result = sp;
        CHECK(correct == result);
      }
      SUBCASE("work cost") {
        float correct = work_cost(g, cost_map);
        float result = work_cost(sp, cost_map);
        CHECK(correct == result);
      }

      SUBCASE("critical path cost") {
        float correct = 32;
        float result = critical_path_cost(sp, cost_map);
        CHECK(correct == result);
      }
    }
  }

  TEST_CASE("cost_aware_stratum_sync_sp_ization") {

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
          cost_aware_stratum_sync_sp_ization(g, cost_map);

      SUBCASE("structure") {
        SerialParallelDecomposition correct(
            SerialSplit{n[0], n[1], ParallelSplit{n[2], n[3]}, n[4], n[5]});
        SerialParallelDecomposition result = sp;
        CHECK(correct == result);
      }
      SUBCASE("work cost") {
        float correct = work_cost(g, cost_map);
        float result = work_cost(sp, cost_map);
        CHECK(correct == result);
      }

      SUBCASE("critical path cost") {
        float correct = 7;
        float result = critical_path_cost(sp, cost_map);
        CHECK(correct == result);
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
          cost_aware_stratum_sync_sp_ization(g, cost_map);

      SUBCASE("structure") {
        SerialParallelDecomposition correct(SerialSplit{
            n[0], ParallelSplit{SerialSplit{n[1], n[3], n[4]}, n[2]}, n[5]});
        SerialParallelDecomposition result = sp;
        CHECK(correct == result);
      }
      SUBCASE("work cost") {
        float correct = work_cost(g, cost_map);
        float result = work_cost(sp, cost_map);
        CHECK(correct == result);
      }

      SUBCASE("critical path cost") {
        float correct = 12;
        float result = critical_path_cost(sp, cost_map);
        CHECK(correct == result);
      }
    }

    SUBCASE("Sample Graph #3") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 9);
      add_edges(g,
                {
                    DirectedEdge{n[0], n[1]},
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
                });

      std::unordered_map<Node, float> cost_map = {{n[0], 1},
                                                  {n[1], 1},
                                                  {n[2], 4},
                                                  {n[3], 10},
                                                  {n[4], 10},
                                                  {n[5], 5},
                                                  {n[6], 4},
                                                  {n[7], 3},
                                                  {n[8], 4}};

      CHECK(work_cost(g, cost_map) == 42);
      CHECK(critical_path_cost(g, cost_map) == 25);

      SerialParallelDecomposition sp =
          cost_aware_stratum_sync_sp_ization(g, cost_map);

      SUBCASE("structure") {
        SerialParallelDecomposition correct(
            SerialSplit{n[0],
                        ParallelSplit{SerialSplit{n[1], n[2], n[6]}, n[3]},
                        ParallelSplit{n[4], SerialSplit{n[5], n[7]}},
                        n[8]});
        SerialParallelDecomposition result = sp;
        CHECK(correct == result);
      }
      SUBCASE("work cost") {
        float correct = work_cost(g, cost_map);
        float result = work_cost(sp, cost_map);
        CHECK(correct == result);
      }

      SUBCASE("critical path cost") {
        float correct = 25;
        float result = critical_path_cost(sp, cost_map);
        CHECK(correct == result);
      }
    }

    SUBCASE("Sample Graph #4") {

      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 15);
      add_edges(g,
                {
                    DirectedEdge{n[0], n[1]},   DirectedEdge{n[0], n[5]},
                    DirectedEdge{n[1], n[2]},   DirectedEdge{n[1], n[3]},
                    DirectedEdge{n[2], n[4]},   DirectedEdge{n[2], n[7]},
                    DirectedEdge{n[3], n[4]},   DirectedEdge{n[3], n[6]},
                    DirectedEdge{n[4], n[6]},   DirectedEdge{n[5], n[6]},
                    DirectedEdge{n[5], n[10]},  DirectedEdge{n[6], n[9]},
                    DirectedEdge{n[6], n[11]},  DirectedEdge{n[7], n[8]},
                    DirectedEdge{n[8], n[9]},   DirectedEdge{n[8], n[13]},
                    DirectedEdge{n[9], n[13]},  DirectedEdge{n[10], n[11]},
                    DirectedEdge{n[10], n[12]}, DirectedEdge{n[11], n[14]},
                    DirectedEdge{n[12], n[14]}, DirectedEdge{n[13], n[14]},
                });

      std::unordered_map<Node, float> cost_map = {{n[0], 1},
                                                  {n[1], 1},
                                                  {n[2], 3},
                                                  {n[3], 3},
                                                  {n[4], 1},
                                                  {n[5], 5},
                                                  {n[6], 5},
                                                  {n[7], 1},
                                                  {n[8], 1},
                                                  {n[9], 1},
                                                  {n[10], 3},
                                                  {n[11], 3},
                                                  {n[12], 2},
                                                  {n[13], 1},
                                                  {n[14], 10}};

      CHECK(work_cost(g, cost_map) == 41);
      CHECK(critical_path_cost(g, cost_map) == 24);

      SerialParallelDecomposition sp =
          cost_aware_stratum_sync_sp_ization(g, cost_map);

      SUBCASE("structure") {
        SerialParallelDecomposition correct(SerialSplit{
            n[0],
            ParallelSplit{SerialSplit{n[1], ParallelSplit{n[2], n[3]}, n[7]},
                          n[5]},
            ParallelSplit{n[4], n[8], n[10]},
            ParallelSplit{n[6], n[12]},
            ParallelSplit{n[11], SerialSplit{n[9], n[13]}},
            n[14]});
        SerialParallelDecomposition result = sp;
        CHECK(correct == result);
      }
      SUBCASE("work cost") {
        float correct = work_cost(g, cost_map);
        float result = work_cost(sp, cost_map);
        CHECK(correct == result);
      }

      SUBCASE("critical path cost") {
        float correct = 27;
        float result = critical_path_cost(sp, cost_map);
        CHECK(correct == result);
      }
    }
  }
}
