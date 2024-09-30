#include "compiler/machine_mapping/get_optimal_machine_mapping.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_tensor_set_movement.h"
#include "compiler/machine_mapping/get_tensor_set_movement_across_split.h"
#include "compiler/machine_mapping/get_allowed_machine_views_list.h"
#include "compiler/machine_mapping/get_machine_resource_splits.h"
#include "compiler/machine_mapping/machine_mapping_constraints.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/machine_mapping_problem_tree.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/mm_problem_tree_parallel_split.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/mm_problem_tree_series_split.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/unmapped_op_cost_estimate_key.h"
#include "compiler/machine_mapping/machine_mapping_result.h"
#include "compiler/machine_mapping/machine_mapping_result_tree/machine_mapping_result_tree.h"
#include "compiler/machine_mapping/mm_problem_tree_series_split.h"
#include "compiler/machine_mapping/partial_machine_mapping.dtg.h"
#include "compiler/machine_mapping/partial_machine_mapping.h"
#include "compiler/machine_mapping/transitive_reduced_pcg.h"
#include "compiler/series_parallel/pcg_binary_sp_decomposition.dtg.h"
#include "compiler/series_parallel/pcg_binary_sp_decomposition.h"
#include "pcg/machine_specification.dtg.h"
#include "pcg/machine_specification.h"
#include "pcg/machine_view.dtg.h"
#include "pcg/machine_view.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "utils/containers/contains.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/get_all_assignments.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/exception.h"
#include "utils/overload.h"
#include "compiler/series_parallel/pcg_binary_parallel_split.dtg.h"
#include "compiler/series_parallel/pcg_binary_parallel_split.h"
#include "compiler/series_parallel/pcg_binary_series_split.h"
#include "compiler/machine_mapping/machine_mapping_context.h"
#include "utils/containers/flatmap.h"
#include "compiler/machine_mapping/estimate_layer_cost.h"


namespace FlexFlow {

MachineMappingResult get_optimal_machine_mapping(
    MachineMappingCache &result_cache,
    MachineMappingContext const &context,
    MachineMappingProblemTree const &problem_tree,
    MachineSpecification const &resources,
    MachineMappingConstraints const &constraints) {

  MachineMappingState state = MachineMappingState{
    problem_tree, resources, constraints,
  };

  {
    std::optional<MachineMappingResult> cached_result =
        result_cache.load(state);
    if (cached_result) {
      return cached_result.value();
    }
  }

  MachineMappingResult result = visit<MachineMappingResult>(
    problem_tree,
    [&](auto const &decomp_tree_node) {
      return get_optimal_machine_mapping
        (result_cache, 
         context, 
         decomp_tree_node, 
         resources, 
         constraints);
    });

  result_cache.save(state, result);
  return result;
}

MachineMappingResult get_optimal_machine_mapping(
    MachineMappingCache &result_cache,
    MachineMappingContext const &context,
    MMProblemTreeSeriesSplit const &series_split,
    MachineSpecification const &resource,
    MachineMappingConstraints const &partial_solution,
    std::optional<ParallelSplitTransformation> const &parallel_split_transformation) {

  MachineMappingResult result = infeasible_machine_mapping_result();

  AbstractedTensorSetMovement tensor_movement = get_abstracted_tensor_movement(series_split);

  auto get_boundary_machine_view_assignments = [&](std::unordered_set<BinaryTreePath> const &layers) 
    -> std::unordered_set<MachineMappingConstraints>
  {
    std::unordered_map<BinaryTreePath, std::unordered_set<MachineView>>
      allowed = generate_map(layers,
                             [&](BinaryTreePath const &l) { 
                               return get_allowed_machine_views_for_layer(context, l);
                             });
    return transform(get_all_assignments(allowed),
                     [](std::unordered_map<parallel_layer_guid_t, MachineView> const &m) {
                       return MachineMapping{m};
                     });
  };

  for (MachineMapping const &assigned_pre_machine_views
        : get_boundary_machine_view_assignments(get_src_layers(tensor_movement))) {

    MachineMappingConstraints pre_candidate = 
      with_additional_constraints(
        restrict_domain(partial_solution, BinaryTreePathEntry::LEFT_CHILD),
        assigned_pre_machine_views);

    MachineMappingResult pre_result = ({
      std::optional<MachineMappingResult> returned
        = get_optimal_machine_mapping(result_cache, 
                                               context, 
                                               get_pre_child(series_split),
                                               resource, 
                                               pre_candidate);
      if (!returned.has_value()) {
        continue;
      }
      returned.value();
    });

    for (MachineMapping const &assigned_post_machine_views
          : get_boundary_machine_view_assignments(get_dst_layers(tensor_movement))) {

      MachineMappingConstraints post_candidate = 
        with_additional_constraints(
          restrict_domain(partial_solution, BinaryTreePathEntry::RIGHT_CHILD),
          assigned_post_machine_views);

      MachineMappingResult post_result = ({
        std::optional<MachineMappingResult> returned 
          = get_optimal_machine_mapping(result_cache, 
                                                 context, 
                                                 get_post_child(series_split),
                                                 resource, 
                                                 post_candidate);
        if (!returned.has_value()) {
          continue;
        }
        returned.value();
      });

      TensorSetMovement comm_across_split = concretize_abstracted_tensor_set_movement(tensor_movement,
                                                                                      /*pre_mapping=*/assigned_pre_machine_views,
                                                                                      /*post_mapping=*/assigned_post_machine_views);
      float cost_across_split = context.cost_estimator.estimate_cost(comm_across_split);

      result = minimize_runtime(result, 
                                series_combine(cost_across_split, pre_result, post_result, parallel_split_transformation));
    }
  }

  return result;
}



MachineMappingResult get_optimal_machine_mapping(
    MachineMappingCache &result_cache,                                                          
    MachineMappingContext const &context,
    MMProblemTreeParallelSplit const &parallel_split,
    MachineSpecification const &resources,
    MachineMappingConstraints const &constraints) {

  MachineMappingProblemTree lhs = get_lhs_child(parallel_split);
  MachineMappingProblemTree rhs = get_rhs_child(parallel_split);

  MachineMappingResult optimal_result = [&] {
    MMProblemTreeSeriesSplit series_split = require_series_split(\
      mm_problem_tree_make_series_split(
        /*tensor_set_movement=*/empty_abstracted_tensor_set_movement(),
        /*pre=*/lhs,
        /*post=*/rhs));
        
    return get_optimal_machine_mapping(result_cache,
                                                context,
                                                series_split,
                                                resources,
                                                constraints,
                                                ParallelSplitTransformation::LthenR);
  }();

  MachineMappingConstraints left_constraints = restrict_domain(constraints, get_leaves(lhs));
  MachineMappingConstraints right_constraints = restrict_domain(constraints, get_leaves(rhs));

  for (auto const &resource_split : get_machine_resource_splits(resources)) {
    MachineMappingResult left_result = 
      get_optimal_machine_mapping(result_cache,
                                           context,
                                           lhs,
                                           resource_split.first,
                                           left_constraints);
    MachineMappingResult right_result = 
      get_optimal_machine_mapping(result_cache,
                                           context,
                                           rhs,
                                           resource_split.second,
                                           right_constraints);

    optimal_result = minimize_runtime(
        optimal_result,
        parallel_combine(left_result, right_result));
  }

  return optimal_result;
}

MachineMappingResult get_optimal_machine_mapping(
    MachineMappingCache &result_cache,
    MachineMappingContext const &context,
    UnmappedOpCostEstimateKey const &leaf,
    MachineSpecification const &resource,
    MachineMappingConstraints const &constraints) {

  MachineView machine_view = require_only_root(constraints).value();
  OpCostEstimateKey mapped = map_unmapped_op_cost_estimate_key(leaf, machine_view);
  float cost = context.cost_estimator.estimate_cost(mapped);

  return make_singleton_machine_mapping_result
    (/*runtime=*/cost,
     /*machine_view=*/machine_view);
}

} // namespace FlexFlow
