#include "compiler/compiler.h"
#include "compiler/cost_estimator/runtime_only_cost_estimator_from_cost_estimator.h"
#include "compiler/mcmc/mcmc_over_mapped_pcg.h"
#include "compiler/search_result.h"
#include "compiler/unity_algorithm/unity_algorithm.h"
#include "pcg/pcg_from_computation_graph.h"
#include "substitutions/unity_substitution_set.h"
#include "utils/overload.h"

namespace FlexFlow {

SearchResult optimize(ComputationGraph const &computation_graph,
                      MachineSpecification const &machine_specification,
                      CostEstimator const &cost_estimator,
                      AlgorithmConfig const &search_config) {
  return search_config.visit<SearchResult>(overload{
      [&](DataParallelismConfig const &config) -> SearchResult {
        throw std::runtime_error(
            "Data parallel search algorithm is not implemented yet");
      },
      [&](UnitySearchConfig const &config) {
        ParallelComputationGraph pcg =
            pcg_from_computation_graph(computation_graph);

        std::vector<Substitution> substitution_set =
            get_expanded_substitution_set(
                machine_specification.compute_specification);

        return optimize_with_unity_algorithm(
            pcg,
            runtime_only_cost_estimator_from_cost_estimator(cost_estimator),
            machine_specification.compute_specification,
            config,
            substitution_set);
      },
      [&](MCMCOverMappedPCGConfig const &config) {
        MachineSpaceCoordinate default_device = MachineSpaceCoordinate{
            /*node_idx=*/0_n,
            /*device_idx=*/0_n,
        };

        SearchResult lifted =
            trivial_search_result_for_cg(computation_graph, default_device);

        ParallelComputationGraph initial_pcg = lifted.pcg;
        MachineMapping initial_mapping = lifted.machine_mapping;

        std::vector<Substitution> substitution_set =
            get_expanded_substitution_set(
                machine_specification.compute_specification);

        return mcmc_over_mapped_pcg(
            initial_pcg,
            runtime_only_cost_estimator_from_cost_estimator(cost_estimator),
            machine_specification,
            config,
            substitution_set,
            initial_mapping);
      },
  });
}

} // namespace FlexFlow
