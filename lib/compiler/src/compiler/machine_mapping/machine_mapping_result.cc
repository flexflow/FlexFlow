#include "compiler/machine_mapping/machine_mapping_result.h"
#include "compiler/cost_estimator/cost_metric.h"
#include "compiler/machine_mapping/machine_mapping.h"
#include "compiler/machine_mapping/parallel_layer_guid_oblivious_machine_mapping.h"
#include "utils/containers/map_keys.h"
#include "utils/containers/merge_maps.h"
#include "utils/full_binary_tree/binary_tree_path.h"

namespace FlexFlow {

MachineMappingResult infeasible_machine_mapping_result() {
  return MachineMappingResult{std::nullopt};
}

bool is_infeasible(MachineMappingResult const &result) {
  return !result.raw_result.has_value();
}

FeasibleMachineMappingResult
    require_feasible(MachineMappingResult const &result) {
  return result.raw_result.value();
}

[[nodiscard]] MachineMappingResult get_mapping_with_minimal_runtime(
    std::unordered_set<MachineMappingResult> const &candidates) {
  MachineMappingResult result = infeasible_machine_mapping_result();

  for (MachineMappingResult const &candidate : candidates) {
    result = minimize_runtime(result, candidate);
  }

  return result;
}

MachineMappingResult
    series_combine(MachineMappingConfig const &config,
                   MachineMemoryConstraints const &memory_constraints,
                   CostMetric const &comm_cost,
                   MachineMappingResult const &maybe_pre_result,
                   MachineMappingResult const &maybe_post_result,
                   std::optional<ParallelSplitTransformation> const
                       &parallel_split_transformation) {
  FeasibleMachineMappingResult pre_result = ({
    if (is_infeasible(maybe_pre_result)) {
      return infeasible_machine_mapping_result();
    }
    require_feasible(maybe_pre_result);
  });

  FeasibleMachineMappingResult post_result = ({
    if (is_infeasible(maybe_post_result)) {
      return infeasible_machine_mapping_result();
    }
    require_feasible(maybe_post_result);
  });

  ParallelLayerGuidObliviousMachineMapping mapping = [&] {
    if (parallel_split_transformation.has_value() &&
        parallel_split_transformation.value() ==
            ParallelSplitTransformation::RthenL) {
      return binary_combine_mappings(/*lhs=*/post_result.machine_mapping,
                                     /*rhs=*/pre_result.machine_mapping);
    } else {
      return binary_combine_mappings(/*lhs=*/pre_result.machine_mapping,
                                     /*rhs=*/post_result.machine_mapping);
    }
  }();

  MachineMappingResult result_without_memory_check = MachineMappingResult{
      FeasibleMachineMappingResult{
          /*cost=*/combine_cost_metrics_inter_device(
              {pre_result.cost, comm_cost, post_result.cost}),
          /*machine_mapping=*/mapping,
      },
  };

  if (config.enable_memory_optimization) {
    return machine_mapping_memory_check(memory_constraints,
                                        result_without_memory_check);
  } else {
    return result_without_memory_check;
  }
}

MachineMappingResult
    parallel_combine(MachineMappingConfig const &config,
                     MachineMemoryConstraints const &memory_constraints,
                     MachineMappingResult const &maybe_lhs_result,
                     MachineMappingResult const &maybe_rhs_result) {
  FeasibleMachineMappingResult lhs_result = ({
    if (is_infeasible(maybe_lhs_result)) {
      return infeasible_machine_mapping_result();
    }
    require_feasible(maybe_lhs_result);
  });

  FeasibleMachineMappingResult rhs_result = ({
    if (is_infeasible(maybe_rhs_result)) {
      return infeasible_machine_mapping_result();
    }
    require_feasible(maybe_rhs_result);
  });

  MachineMappingResult result_without_memory_check = MachineMappingResult{
      FeasibleMachineMappingResult{
          /*cost=*/combine_cost_metrics_intra_device_parallel(lhs_result.cost,
                                                              rhs_result.cost),
          /*machine_mapping=*/
          binary_combine_mappings(/*lhs=*/lhs_result.machine_mapping,
                                  /*rhs=*/rhs_result.machine_mapping),
      },
  };

  if (config.enable_memory_optimization) {
    return machine_mapping_memory_check(memory_constraints,
                                        result_without_memory_check);
  } else {
    return result_without_memory_check;
  }
}

MachineMappingResult minimize_runtime(MachineMappingResult const &maybe_m1,
                                      MachineMappingResult const &maybe_m2) {
  FeasibleMachineMappingResult m1 = ({
    if (is_infeasible(maybe_m1)) {
      return maybe_m2;
    }
    require_feasible(maybe_m1);
  });

  FeasibleMachineMappingResult m2 = ({
    if (is_infeasible(maybe_m2)) {
      return maybe_m1;
    }
    require_feasible(maybe_m2);
  });

  if (m2.cost.runtime < m1.cost.runtime) {
    return maybe_m2;
  } else {
    return maybe_m1;
  }
}

MachineMappingResult make_singleton_machine_mapping_result(
    MachineMappingConfig const &config,
    MachineMemoryConstraints const &memory_constraints,
    CostMetric const &cost,
    MachineView const &machine_view) {
  MachineMappingResult result_without_memory_check = MachineMappingResult{
      FeasibleMachineMappingResult{
          /*cost=*/cost,
          /*machine_mapping=*/
          ParallelLayerGuidObliviousMachineMapping{{
              {binary_tree_root_path(), machine_view},
          }},
      },
  };

  return machine_mapping_memory_check(memory_constraints,
                                      result_without_memory_check);
}

MachineMappingResult machine_mapping_memory_check(
    MachineMemoryConstraints const &memory_constraints,
    MachineMappingResult const &result) {
  FeasibleMachineMappingResult feasible_result = ({
    if (is_infeasible(result)) {
      return infeasible_machine_mapping_result();
    }
    require_feasible(result);
  });

  if (feasible_result.cost.memory > memory_constraints.memory_limit) {
    return infeasible_machine_mapping_result();
  } else {
    return result;
  }
}

} // namespace FlexFlow
