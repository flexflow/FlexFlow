#include "compiler/machine_mapping/machine_mapping_result.h"
#include "compiler/machine_mapping/machine_mapping.h"
#include "utils/containers/map_keys.h"
#include "utils/containers/merge_maps.h"
#include "utils/full_binary_tree/binary_tree_path.h"

namespace FlexFlow {

MachineMappingResult sequential_combine(float comm_cost,
                                        MachineMappingResult const &maybe_pre_result,
                                        MachineMappingResult const &maybe_post_result,
                                        std::optional<ParallelSplitTransformation> const &parallel_split_transformation) {
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

  std::function<BinaryTreePath(BinaryTreePath const &)> transform_problem_tree_paths_pre 
    = nest_inside_left_child;
  std::function<BinaryTreePath(BinaryTreePath const &)> transform_problem_tree_paths_post
    = nest_inside_right_child;

  if (parallel_split_transformation.has_value() 
      && parallel_split_transformation.value() == ParallelSplitTransformation::RthenL) {
    transform_problem_tree_paths_pre = nest_inside_right_child;
    transform_problem_tree_paths_post = nest_inside_left_child;
  }

  return MachineMappingResult{
    FeasibleMachineMappingResult{
      /*runtime=*/pre_result.runtime + comm_cost + post_result.runtime,
      /*parallel_layer_guid_oblivious_machine_mapping=*/merge_maps(
        map_keys(pre_result.parallel_layer_guid_oblivious_machine_mapping,
                 transform_problem_tree_paths_pre),
        map_keys(post_result.parallel_layer_guid_oblivious_machine_mapping,
                 transform_problem_tree_paths_post)),
    },
  };
}

MachineMappingResult parallel_combine(MachineMappingResult const &maybe_lhs_result,
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

  return MachineMappingResult{
    FeasibleMachineMappingResult{
      /*runtime=*/std::max(lhs_result.runtime, rhs_result.runtime),
      /*parallel_layer_guid_oblivious_machine_mapping=*/merge_maps(
        map_keys(lhs_result.parallel_layer_guid_oblivious_machine_mapping,
                 nest_inside_left_child),
        map_keys(rhs_result.parallel_layer_guid_oblivious_machine_mapping,
                 nest_inside_right_child)),
    },
  };
}

MachineMappingResult infeasible_machine_mapping_result() {
  return MachineMappingResult{std::nullopt};
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

  if (m2.runtime < m1.runtime) {
    return maybe_m2;
  } else {
    return maybe_m1;
  }
}

MachineMappingResult make_singleton_machine_mapping_result(float runtime,
                                                           MachineView const &machine_view) {
  return MachineMappingResult{
    FeasibleMachineMappingResult{
      /*runtime=*/runtime,
      /*parallel_layer_guid_oblivious_machine_mapping=*/{
        {binary_tree_root_path(), machine_view},
      },
    },
  };
}

} // namespace FlexFlow
