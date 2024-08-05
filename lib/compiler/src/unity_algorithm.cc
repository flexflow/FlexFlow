#include "compiler/unity_algorithm.h"
#include "compiler/machine_mapping.h"
#include "pcg/machine_specification.dtg.h"
#include "substitutions/substitution.h"
#include "utils/deduplicated_priority_queue.h"
#include "utils/graph/node/algorithms.h"
namespace FlexFlow {

bool StrategyRuntimeCmp::operator()(Strategy const &lhs, Strategy const &rhs) {
  return lhs.runtime < rhs.runtime;
}

/*
 * Gets all substitutions applicable to a PCG
 */
std::unordered_set<Substitution>
    get_all_applicable_substitutions(ParallelComputationGraph const &pcg) {
  NOT_IMPLEMENTED();
}

/*
 * Applies a substitution to all possible positions in PCG
 */
std::unordered_set<ParallelComputationGraph>
    apply_substitution(ParallelComputationGraph const &pcg,
                       Substitution const &) {
  NOT_IMPLEMENTED();
}

Strategy graph_optimize(
    ComputationGraph &cg,
    CostEstimator const &cost_estimator,
    MachineSpecification const &resources,
    std::function<std::unordered_set<MachineView>(
        ParallelLayerAttrs const &, MachineSpecification const &)> const
        &allowed_machine_views,
    OptimizerConfig const &opt_config) {
  NOT_IMPLEMENTED();
  // ParallelComputationGraph pcg = cg_to_pcg(cg);

  // std::unordered_set<Substitution> subs =
  // get_all_applicable_substitutions(pcg);

  // OptimalCostCache cached_subgraph_costs;
  // DeduplicatedPriorityQueue<Strategy, std::vector<Strategy>,
  // StrategyRuntimeCmp>
  //     candidates;

  // OptimalCostResult initial_pcg_result = optimal_cost(pcg,
  //                                                     allowed_machine_views,
  //                                                     cost_estimator,
  //                                                     resources,
  //                                                     cached_subgraph_costs);
  // Strategy initial_result{
  //     pcg, initial_pcg_result.machine_mapping, initial_pcg_result.runtime};

  // Strategy best_result = initial_result;
  // candidates.push(initial_result);

  // for (int iteration = 0; !candidates.empty() && iteration <
  // opt_config.budget;
  //      ++iteration) {
  //   Strategy const &current_result = candidates.top();
  //   candidates.pop();

  //   if (current_result.runtime < best_result.runtime) {
  //     best_result = current_result;
  //   } else if (current_result.runtime >
  //              best_result.runtime * opt_config.alpha) {
  //     continue;
  //   }

  //   for (auto const &sub : subs) {
  //     for (auto const &new_pcg : apply_substitution(current_result.pcg, sub))
  //     {
  //       OptimalCostResult c = optimal_cost(new_pcg,
  //                                          allowed_machine_views,
  //                                          cost_estimator,
  //                                          resources,
  //                                          cached_subgraph_costs);
  //       Strategy new_result{new_pcg, c.machine_mapping, c.runtime};
  //       if (new_result.runtime <= opt_config.threshold &&
  //           get_nodes(new_pcg.raw_graph).size() <= opt_config.max_num_ops) {
  //         candidates.push(new_result);
  //       }
  //     }
  //   }
  // }

  // return best_result;
}

} // namespace FlexFlow
