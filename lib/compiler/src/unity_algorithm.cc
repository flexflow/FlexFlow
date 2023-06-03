#include "compiler/unity_algorithm.h"
#include "deduplicated_priority_queue.h"
#include "graph_utils.h"
#include "substitutions_implementation.h"

namespace FlexFlow {

Strategy::Strategy(OptimizerPCG const &pcg, MachineMapping const &machine_mapping)
    : pcg(pcg), machine_mapping(machine_mapping) {}

bool StrategyRuntimeCmp::operator()(Strategy const &lhs, Strategy const &rhs) {
  return lhs.machine_mapping.runtime < rhs.machine_mapping.runtime;
}

std::unordered_set<Substitution> get_all_substitutions(OptimizerPCG const &pcg);

std::unordered_set<OptimizerPCG> apply_substitution(OptimizerPCG const &pcg,
                                                    Substitution const &);

Strategy graph_optimize(
    OptimizerComputationGraph &cg,
    ICostEstimator const &cost_estimator,
    MachineSpecification const &resources,
    std::function<std::unordered_set<MachineView>(
        PCGOperatorAttrs const &, MachineSpecification const &)> const
        &allowed_machine_views,
    OptimizerConfig const &opt_config) {

  OptimizerPCG pcg = cg_to_pcg(cg);

  std::unordered_set<Substitution> subs = get_all_substitutions(pcg);

  std::unordered_map<size_t, MachineMapping> cached_subgraph_costs;
  DeduplicatedPriorityQueue<Strategy, std::vector<Strategy>, StrategyRuntimeCmp>
      candidates;

  Strategy initial_result(pcg,
                          optimal_cost(pcg,
                                       allowed_machine_views,
                                       cost_estimator,
                                       resources,
                                       cached_subgraph_costs));

  Strategy best_result = initial_result;
  candidates.push(initial_result);

  for (int iteration = 0; !candidates.empty() && iteration < opt_config.budget;
       ++iteration) {
    Strategy const &current_result = candidates.top();
    candidates.pop();

    if (StrategyRuntimeCmp(current_result, best_result)) {
      best_result = current_result;
    } else if (current_result.machine_mapping.runtime >
               best_result.machine_mapping.runtime * opt_config.alpha) {
      continue;
    }

    for (auto const &sub : subs) {
      for (auto const &new_pcg : apply_substitution(current_result.pcg, sub)) {
        Strategy new_result(new_pcg,
                            optimal_cost(new_pcg,
                                         allowed_machine_views,
                                         cost_estimator,
                                         resources,
                                         cached_subgraph_costs));
        if (new_result.machine_mapping.runtime <= opt_config.threshold &&
            new_result.pcg.query_nodes({}).size() <= opt_config.max_num_ops) {
          candidates.push(new_result);
        }
      }
    }
  }

  return best_result;
}

}

namespace std {
  
size_t hash<::FlexFlow::Strategy>::operator()(::FlexFlow::Strategy const &s) const {
  size_t h = hash<FlexFlow::OptimizerPCG>{}(s.pcg);
  hash_combine(h, s.machine_mapping);
  return h;
}

}
