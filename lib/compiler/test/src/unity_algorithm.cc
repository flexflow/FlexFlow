#include "compiler/unity_algorithm.h"
#include "doctest/doctest.h"

TEST_SUITE(FF_TEST_SUITE) {
  // Rapidcheck does not work for now
  // TEST_CASE("graph_optimize") {
  //   RC_SUBCASE([](ComputationGraph const &g,
  //                float alpha,
  //                int budget,
  //                float threshold,
  //                int max_num_ops) {
  //     Strategy s = graph_optimize(
  //         g,
  //         TestCostEstimator{},
  //         MachineSpecification{1, 1, 4, 0.1, 0.2},
  //         [](Operator const &, MachineSpecification const &) {
  //           return std::unordered_set<MachineView>{make_1d_machine_view(0, 1,
  //           1)};
  //         },
  //         OptimizerConfig{alpha, budget, threshold, max_num_ops});
  //     RC_ASSERT(get_nodes(s.pcg).size() > 0);
  //     RC_ASSERT(s.machine_mapping.runtime > 0);
  //     RC_ASSERT(keys(s.machine_mapping.machine_views) == get_nodes(s.pcg));
  //   });
  // }
}
