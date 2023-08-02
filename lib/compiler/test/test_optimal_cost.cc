#include "test_cost_estimator.h"
#include "test_generator.h"

TEST_CASE("optimal_cost") {
  rc::check([](ParallelComputationGraph const &g,
               MachineSpecification const &machine_spec) {
    std::unordered_map<size_t, MachineMapping> cached_subgraph_costs;
    MachineMapping machine_mapping = optimal_cost(
        g,
        [](Operator const &, MachineSpecification const &) {
          return std::unordered_set<MachineView>{make_1d_machine_view(0, 1, 1)};
        },
        TestCostEstimator{},
        machine_spec,
        cached_subgraph_costs);
    RC_ASSERT(machine_mapping.runtime > 0);
    RC_ASSERT(keys(machine_mapping.machine_views) == get_nodes(g));
  });
}