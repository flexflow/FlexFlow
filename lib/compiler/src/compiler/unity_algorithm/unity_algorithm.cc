#include "compiler/unity_algorithm/unity_algorithm.h"
#include "compiler/machine_mapping/allowed_machine_views.h"
#include "compiler/machine_mapping/get_optimal_machine_mapping.h"
#include "compiler/machine_mapping/machine_mapping.h"
#include "compiler/machine_mapping/machine_mapping_cache.h"
#include "compiler/machine_mapping/machine_mapping_constraints.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/get_machine_mapping_problem_tree.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/machine_mapping_problem_tree.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/unmapped_op_cost_estimate_key.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/unmapped_runtime_only_op_cost_estimate_key.h"
#include "compiler/machine_mapping/machine_mapping_result.h"
#include "compiler/series_parallel/pcg/get_pcg_balanced_binary_sp_decomposition.h"
#include "compiler/series_parallel/pcg/get_pcg_series_parallel_decomposition.h"
#include "compiler/unity_algorithm/graph_optimize_state.h"
#include "op-attrs/operator_task_space.h"
#include "pcg/machine_compute_resource_slice.h"
#include "pcg/machine_specification.dtg.h"
#include "substitutions/apply_substitution/apply_substitution.h"
#include "substitutions/pcg_pattern.h"
#include "substitutions/sub_parallel_computation_graph.h"
#include "substitutions/substitution.h"
#include "substitutions/unity_substitution_set.h"
#include "utils/containers/generate_map.h"
#include "utils/deduplicated_priority_queue.h"
#include "utils/graph/node/algorithms.h"
#include "utils/optional.h"

namespace FlexFlow {

/*
 * Applies a substitution to all possible positions in PCG
 */
//! [apply_substitution example]
std::vector<ParallelComputationGraph>
    all_pcgs_obtained_by_applying_a_substitution(
        ParallelComputationGraph const &pcg,
        std::vector<Substitution> const &substitutions) {
  std::vector<ParallelComputationGraph> results;
  SubParallelComputationGraph subpcg = sub_pcg_from_full_pcg(pcg);
  for (Substitution const &substitution : substitutions) {
    for (PCGPatternMatch const &pattern_match :
         find_pattern_matches(substitution.pcg_pattern, subpcg)) {
      SubParallelComputationGraph subpcg_from_substitution =
          apply_substitution(subpcg, substitution, pattern_match);
      results.push_back(
          pcg_from_sub_pcg_by_dropping_inputs(subpcg_from_substitution));
    }
  }
  return results;
}
//! [apply_substitution example]

SearchResult optimize_with_unity_algorithm(
    ParallelComputationGraph &pcg,
    RuntimeOnlyCostEstimator const &cost_estimator,
    MachineComputeSpecification const &resources,
    UnitySearchConfig const &search_config,
    std::vector<Substitution> const &substitutions) {

  MachineMappingCache cached_subgraph_costs = empty_machine_mapping_cache();
  DeduplicatedPriorityQueue<GraphOptimizeState> candidates;

  MachineMappingContext context = MachineMappingContext{
      /*cost_estimator=*/cost_estimator,
      /*allowed_machine_views=*/
      [&](UnmappedRuntimeOnlyOpCostEstimateKey const &key,
          MachineComputeResourceSlice const &resources)
          -> std::set<MachineView> {
        OperatorTaskSpace op_task_space =
            get_operator_task_space_for_runtime_only_op_cost_estimate_key(key);

        return get_allowed_machine_views(resources, op_task_space);
      },
  };

  auto optimize_pcg = [&](ParallelComputationGraph const &pcg)
      -> std::pair<GraphOptimizeState, std::optional<MachineMapping>> {
    PCGBinarySPDecomposition sp_decomp =
        expect(get_pcg_balanced_binary_sp_decomposition(pcg),
               "Failed to get SP decomposition of PCG");

    MachineMappingProblemTree problem_tree =
        get_machine_mapping_problem_tree(pcg, sp_decomp);
    MachineMappingConstraints constraints =
        get_unconstrained_solution_for_layers(get_all_leaf_paths(problem_tree));

    MachineMappingResult mm_result =
        get_optimal_machine_mapping(cached_subgraph_costs,
                                    context,
                                    problem_tree,
                                    compute_slice_from_specification(resources),
                                    constraints);

    return {
        GraphOptimizeState{
            /*pcg=*/pcg,
            /*runtime=*/get_runtime_cost(mm_result),
        },
        get_machine_mapping_from_machine_mapping_result(sp_decomp, mm_result),
    };
  };

  GraphOptimizeState best_state = optimize_pcg(pcg).first;
  candidates.push(best_state);

  for (int iteration = 0;
       !candidates.empty() && iteration < search_config.budget;
       ++iteration) {
    GraphOptimizeState current_state = candidates.top();
    candidates.pop();

    if (current_state < best_state) {
      best_state = current_state;
    } else if (current_state.runtime >
               best_state.runtime * search_config.alpha) {
      continue;
    }

    for (ParallelComputationGraph const &new_pcg :
         all_pcgs_obtained_by_applying_a_substitution(current_state.pcg,
                                                      substitutions)) {

      std::optional<GraphOptimizeState> new_pcg_optimize_result =
          optimize_pcg(new_pcg).first;

      if (new_pcg_optimize_result == std::nullopt) {
        continue;
      }

      GraphOptimizeState new_state = new_pcg_optimize_result.value();
      if (new_state.runtime <= best_state.runtime * search_config.alpha &&
          get_nodes(new_pcg.raw_graph).size() <= search_config.max_num_ops) {
        candidates.push(new_state);
      }
    }
  }

  std::optional<MachineMapping> best_mapping =
      optimize_pcg(best_state.pcg).second;

  ASSERT(best_mapping != std::nullopt, "Failed to find any solutions");

  return SearchResult{
      /*pcg=*/best_state.pcg,
      /*machine_mapping=*/best_mapping.value(),
  };
}

} // namespace FlexFlow
