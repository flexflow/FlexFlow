#include "compiler/unity_algorithm.h"
#include "compiler/graph_optimize_state.h"
#include "compiler/machine_mapping/get_optimal_machine_mapping.h"
#include "pcg/machine_specification.dtg.h"
#include "substitutions/substitution.h"
#include "utils/deduplicated_priority_queue.h"
#include "utils/graph/node/algorithms.h"
namespace FlexFlow {

/*
 * Gets all substitutions applicable to a PCG
 */
std::vector<Substitution>
    get_all_applicable_substitutions(ParallelComputationGraph const &pcg) {
  NOT_IMPLEMENTED();
}

/*
 * Applies a substitution to all possible positions in PCG
 */
std::vector<ParallelComputationGraph>
    apply_substitution(ParallelComputationGraph const &pcg,
                       Substitution const &) {
  NOT_IMPLEMENTED();
}

GraphOptimizeResult graph_optimize(
    ParallelComputationGraph &pcg,
    CostEstimator const &cost_estimator,
    MachineSpecification const &resources,
    std::function<std::unordered_set<MachineView>(
        ParallelLayerAttrs const &, MachineSpecification const &)> const
        &allowed_machine_views,
    OptimizerConfig const &opt_config) {
  std::vector<Substitution> substitutions =
      get_all_applicable_substitutions(pcg);

  MachineMappingCache cached_subgraph_costs;
  DeduplicatedPriorityQueue<GraphOptimizeState> candidates;

  MachineMappingResult original_pcg_cost =
      get_optimal_machine_mapping(pcg,
                                  allowed_machine_views,
                                  cost_estimator,
                                  resources,
                                  cached_subgraph_costs);

  GraphOptimizeState initial_state = {
      GraphOptimizeResult(pcg, original_pcg_cost.machine_mapping),
      original_pcg_cost.runtime};

  GraphOptimizeState best_state = initial_state;
  candidates.push(initial_state);

  for (int iteration = 0; !candidates.empty() && iteration < opt_config.budget;
       ++iteration) {
    GraphOptimizeState current_state = candidates.top();
    candidates.pop();

    if (current_state.runtime < best_state.runtime) {
      best_state = current_state;
    } else if (current_state.runtime > best_state.runtime * opt_config.alpha) {
      continue;
    }

    for (Substitution const &substitution : substitutions) {
      for (ParallelComputationGraph const &new_pcg : apply_substitution(
               current_state.graph_optimize_result.pcg, substitution)) {
        MachineMappingResult new_pcg_cost =
            get_optimal_machine_mapping(new_pcg,
                                        allowed_machine_views,
                                        cost_estimator,
                                        resources,
                                        cached_subgraph_costs);
        GraphOptimizeState new_state{
            GraphOptimizeResult(new_pcg, new_pcg_cost.machine_mapping),
            new_pcg_cost.runtime};
        if (new_pcg_cost.runtime <= opt_config.threshold &&
            get_nodes(new_pcg.raw_graph).size() <= opt_config.max_num_ops) {
          candidates.push(new_state);
        }
      }
    }
  }

  return best_state.graph_optimize_result;
}

} // namespace FlexFlow
