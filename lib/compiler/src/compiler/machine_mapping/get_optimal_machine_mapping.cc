#include "compiler/machine_mapping/get_optimal_machine_mapping.h"
#include "compiler/cost_estimator/op_cost_metrics.dtg.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_tensor_set_movement.h"
#include "compiler/machine_mapping/machine_mapping_cache.h"
#include "compiler/machine_mapping/machine_mapping_constraints.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/machine_mapping_problem_tree.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/unmapped_op_cost_estimate_key.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/unmapped_runtime_only_op_cost_estimate_key.h"
#include "compiler/machine_mapping/machine_mapping_result.h"
#include "compiler/machine_mapping/machine_resource_split.dtg.h"
#include "compiler/machine_mapping/machine_resource_split.h"
#include "compiler/machine_mapping/machine_view.dtg.h"
#include "compiler/machine_mapping/machine_view.h"
#include "compiler/machine_mapping/parallel_layer_guid_oblivious_machine_mapping.h"
#include "compiler/machine_mapping/transitive_reduced_pcg.h"
#include "compiler/series_parallel/pcg/pcg_binary_sp_decomposition.dtg.h"
#include "compiler/series_parallel/pcg/pcg_binary_sp_decomposition.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "op-attrs/get_operator_task_space.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/machine_specification.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "utils/containers/contains.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/flatmap.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/get_all_assignments.h"
#include "utils/containers/keys.h"
#include "utils/containers/set_minus.h"
#include "utils/containers/set_of.h"
#include "utils/overload.h"

namespace FlexFlow {

MachineMappingResult
    get_optimal_machine_mapping(MachineMappingCache &result_cache,
                                MachineMappingContext const &context,
                                MachineMappingProblemTree const &problem_tree,
                                MachineComputeResourceSlice const &resources,
                                MachineMappingConstraints const &constraints) {

  ASSERT(get_all_layers(constraints) == get_all_leaf_paths(problem_tree));

  MachineMappingState state = MachineMappingState{
      problem_tree,
      resources,
      constraints,
  };

  {
    std::optional<MachineMappingResult> cached_result =
        machine_mapping_cache_load(result_cache, state);
    if (cached_result) {
      return cached_result.value();
    }
  }

  MachineMappingResult result =
      problem_tree.visit<MachineMappingResult>(overload{
          [&](MMProblemTreeSeriesSplit const &series_split) {
            return get_optimal_machine_mapping(
                result_cache,
                context,
                series_split,
                resources,
                constraints,
                /*parallel_split_transformation=*/std::nullopt);
          },
          [&](auto const &decomp_tree_node) {
            return get_optimal_machine_mapping(result_cache,
                                               context,
                                               decomp_tree_node,
                                               resources,
                                               constraints);
          },
      });

  machine_mapping_cache_save(result_cache, state, result);
  return result;
}

MachineMappingResult
    get_optimal_machine_mapping(MachineMappingCache &result_cache,
                                MachineMappingContext const &context,
                                MMProblemTreeSeriesSplit const &series_split,
                                MachineComputeResourceSlice const &resources,
                                MachineMappingConstraints const &constraints,
                                std::optional<ParallelSplitTransformation> const
                                    &parallel_split_transformation) {

  auto get_boundary_machine_view_assignments =
      [&](std::set<BinaryTreePath> const &boundary_layers,
          MachineMappingProblemTree const &root,
          BinaryTreePathEntry const &prefix)
      -> std::set<ParallelLayerGuidObliviousMachineMapping> {
    MachineMappingConstraints sub_constraints =
        restrict_to_child(constraints, prefix);

    ASSERT(get_all_layers(sub_constraints) == get_all_leaf_paths(root));

    std::set<BinaryTreePath> unconstrained_boundary_layers = set_minus(
        boundary_layers, set_of(get_constrained_layers(sub_constraints)));

    std::map<BinaryTreePath, std::set<MachineView>> allowed =
        generate_map(unconstrained_boundary_layers,
                     [&](BinaryTreePath const &l) -> std::set<MachineView> {
                       UnmappedRuntimeOnlyOpCostEstimateKey leaf =
                           mm_problem_tree_get_subtree_at_path(root, l)
                               .value()
                               .get<UnmappedRuntimeOnlyOpCostEstimateKey>();
                       return context.allowed_machine_views(leaf, resources);
                     });

    std::set<std::map<BinaryTreePath, MachineView>> assignments =
        get_all_assignments(allowed);

    return transform(assignments,
                     [](std::map<BinaryTreePath, MachineView> const &m) {
                       return ParallelLayerGuidObliviousMachineMapping{m};
                     });
  };

  auto eval_pre_boundary_mapping =
      [&](ParallelLayerGuidObliviousMachineMapping const
              &assigned_pre_machine_views) {
        MachineMappingConstraints pre_candidate = with_additional_constraints(
            restrict_to_left_child(constraints), assigned_pre_machine_views);

        MachineMappingResult pre_result =
            get_optimal_machine_mapping(result_cache,
                                        context,
                                        series_split.get_left_child(),
                                        resources,
                                        pre_candidate);

        return pre_result;
      };

  auto eval_post_boundary_mapping =
      [&](ParallelLayerGuidObliviousMachineMapping const
              &assigned_post_machine_views) {
        MachineMappingConstraints post_candidate = with_additional_constraints(
            restrict_to_right_child(constraints), assigned_post_machine_views);

        MachineMappingResult post_result =
            get_optimal_machine_mapping(result_cache,
                                        context,
                                        series_split.get_right_child(),
                                        resources,
                                        post_candidate);

        return post_result;
      };

  MachineMappingResult result = infeasible_machine_mapping_result();
  AbstractedTensorSetMovement tensor_movement =
      series_split.tensor_set_movement;

  for (ParallelLayerGuidObliviousMachineMapping const
           &assigned_pre_machine_views :
       get_boundary_machine_view_assignments(get_src_layers(tensor_movement),
                                             series_split.get_left_child(),
                                             BinaryTreePathEntry::LEFT_CHILD)) {

    MachineMappingResult pre_result =
        eval_pre_boundary_mapping(assigned_pre_machine_views);

    if (is_infeasible(pre_result)) {
      continue;
    }

    for (ParallelLayerGuidObliviousMachineMapping const
             &assigned_post_machine_views :
         get_boundary_machine_view_assignments(
             get_dst_layers(tensor_movement),
             series_split.get_right_child(),
             BinaryTreePathEntry::RIGHT_CHILD)) {

      MachineMappingResult post_result =
          eval_post_boundary_mapping(assigned_post_machine_views);

      if (is_infeasible(post_result)) {
        continue;
      }

      TensorSetMovement comm_across_split =
          concretize_abstracted_tensor_set_movement(
              tensor_movement,
              /*pre_machine_stencils=*/
              get_machine_stencils_for_partially_mapped_mm_problem_tree(
                  series_split.get_left_child(),
                  require_feasible(pre_result).machine_mapping),
              /*post_machine_stencils=*/
              get_machine_stencils_for_partially_mapped_mm_problem_tree(
                  series_split.get_right_child(),
                  require_feasible(post_result).machine_mapping));

      milliseconds_t cost_across_split =
          context.cost_estimator.estimate_cost(comm_across_split);

      result = minimize_runtime(result,
                                series_combine(cost_across_split,
                                               pre_result,
                                               post_result,
                                               parallel_split_transformation));
    }
  }

  return result;
}

MachineMappingResult get_optimal_machine_mapping(
    MachineMappingCache &result_cache,
    MachineMappingContext const &context,
    MMProblemTreeParallelSplit const &parallel_split,
    MachineComputeResourceSlice const &resources,
    MachineMappingConstraints const &constraints) {

  MachineMappingProblemTree lhs = parallel_split.get_left_child();
  MachineMappingProblemTree rhs = parallel_split.get_right_child();

  MachineMappingResult series_result = [&] {
    MMProblemTreeSeriesSplit series_split = MMProblemTreeSeriesSplit{
        /*tensor_set_movement=*/empty_abstracted_tensor_set_movement(),
        /*left_child=*/lhs,
        /*right_child=*/rhs,
    };

    return get_optimal_machine_mapping(result_cache,
                                       context,
                                       series_split,
                                       resources,
                                       constraints,
                                       ParallelSplitTransformation::LthenR);
  }();

  MachineMappingConstraints left_constraints =
      restrict_to_left_child(constraints);
  MachineMappingConstraints right_constraints =
      restrict_to_right_child(constraints);

  auto evaluate_resource_split =
      [&](MachineResourceSplit const &resource_split) {
        auto [lhs_resources, rhs_resources] =
            apply_resource_split(resource_split, resources);

        MachineMappingResult left_result = get_optimal_machine_mapping(
            result_cache, context, lhs, lhs_resources, left_constraints);
        MachineMappingResult right_result = get_optimal_machine_mapping(
            result_cache, context, rhs, rhs_resources, right_constraints);

        return parallel_combine(resource_split, left_result, right_result);
      };

  std::set<MachineMappingResult> parallel_results = transform(
      get_machine_resource_splits(resources), evaluate_resource_split);

  return minimize_runtime(series_result,
                          get_mapping_with_minimal_runtime(parallel_results));
}

MachineMappingResult get_optimal_machine_mapping(
    MachineMappingCache &result_cache,
    MachineMappingContext const &context,
    UnmappedRuntimeOnlyOpCostEstimateKey const &leaf,
    MachineComputeResourceSlice const &resource,
    MachineMappingConstraints const &constraints) {

  std::set<MachineView> candidates = [&] {
    std::optional<MachineView> machine_view = require_only_root(constraints);
    if (machine_view.has_value()) {
      return std::set{machine_view.value()};
    } else {
      return context.allowed_machine_views(leaf, resource);
    }
  }();

  auto get_mapping_result = [&](MachineView const &machine_view) {
    RuntimeOnlyOpCostEstimateKey mapped =
        map_unmapped_runtime_only_op_cost_estimate_key(leaf, machine_view);
    RuntimeOnlyOpCostMetrics metrics =
        context.cost_estimator.estimate_cost(mapped);
    milliseconds_t cost = metrics.forward_runtime + metrics.backward_runtime;
    return make_singleton_machine_mapping_result(cost, machine_view);
  };

  std::set<MachineMappingResult> candidate_results =
      transform(candidates, get_mapping_result);

  return get_mapping_with_minimal_runtime(candidate_results);
}

} // namespace FlexFlow
