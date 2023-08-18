#include "test_cost_estimator.h"
#include "test_generator.h"

/*
Tests whether optimal_cost can give a valid result given random PCG, trivial
allowed machine views, trivial cost estimator and random machine specification.
*/
TEST_CASE("optimal_cost") {
  auto test_allowed_machine_views = [](Operator const &,
                                       MachineSpecification const &) {
    return std::unordered_set<MachineView>{make_1d_machine_view(0, 1, 1)};
  };
  rc::check([](ParallelComputationGraph const &g,
               MachineSpecification const &machine_spec) {
    OptimalCostCache cached_subgraph_costs;
    OptimalCostResult result = optimal_cost(g,
                                            test_allowed_machine_views,
                                            TestCostEstimator{},
                                            machine_spec,
                                            cached_subgraph_costs);
    RC_ASSERT(result.runtime > 0);
    RC_ASSERT(keys(result.machine_mapping.machine_views) == get_nodes(g));
  });
}
