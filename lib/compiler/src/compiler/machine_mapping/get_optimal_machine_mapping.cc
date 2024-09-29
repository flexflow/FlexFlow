#include "compiler/machine_mapping/get_optimal_machine_mapping.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_tensor_set_movement.h"
#include "compiler/machine_mapping/get_tensor_set_movement_across_split.h"
#include "compiler/machine_mapping/get_allowed_machine_views_list.h"
#include "compiler/machine_mapping/get_machine_resource_splits.h"
#include "compiler/machine_mapping/machine_mapping_constraints.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/machine_mapping_problem_tree.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/mm_problem_tree_series_split.h"
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
    ParallelComputationGraph const &pcg,
    std::function<std::unordered_set<MachineView>(
        ParallelLayerAttrs const &, MachineSpecification const &)> const
        &allowed_machine_views,
    CostEstimator const &cost_estimator,
    MachineSpecification const &resources,
    MachineMappingCache &result_cache) {

  MachineMappingContext context = make_machine_mapping_context(
      pcg, 
      cost_estimator, 
      allowed_machine_views);

  MachineMappingResult result =
      get_optimal_machine_mapping_internal(result_cache, context, resources);

  return result;
}

MachineMappingResultTree get_optimal_machine_mapping_internal(
    MachineMappingCache &result_cache,                                                          
    MachineMappingContext const &context, 
    MachineSpecification const &resources) {

  std::unordered_set<parallel_layer_guid_t> all_layers = get_parallel_layers(context.transitive_reduced_pcg.full_pcg);

  NOT_IMPLEMENTED();
  // return get_optimal_machine_mapping_internal(result_cache, 
  //                                             context, 
  //                                             sp_decomposition_tree, 
  //                                             resources,
  //                                             get_unconstrained_solution_for_layers(all_layers));
}

MachineMappingResultTree get_optimal_machine_mapping_internal(
    MachineMappingCache &result_cache,
    MachineMappingContext const &context,
    MachineMappingProblemTree const &problem_tree,
    MachineSpecification const &resources,
    MachineMappingConstraints const &constraints) {

  MachineMappingState state = MachineMappingState{
    problem_tree, resources, constraints,
  };

  {
    std::optional<MachineMappingResultTree> cached_result =
        result_cache.load(state);
    if (cached_result) {
      return cached_result.value();
    }
  }

  MachineMappingResult result = visit<MachineMappingResult>(
    sp_decomposition_tree,
    [&](auto const &decomp_tree_node) {
      return get_optimal_machine_mapping_internal(result_cache, context, decomp_tree_node, resources, partial_solution);
    });

  result_cache.save(state, result);
  return result;
}

std::optional<MachineMappingResultTree> get_optimal_machine_mapping_internal(
    MachineMappingCache &result_cache,
    MachineMappingContext const &context,
    MMProblemTreeSeriesSplit const &series_split,
    MachineSpecification const &resource,
    MachineMappingConstraints const &partial_solution) {

  std::optional<MachineMappingResultTree> result = std::nullopt;

  auto is_subgraph_input = [&](std::unordered_set<Node> const &subgraph_nodes,
                               parallel_tensor_guid_t const &input_tensor) {
    return !contains(subgraph_nodes, input_tensor.raw_graph_output.node);
  };

  AbstractedTensorSetMovement tensor_movement = get_abstracted_tensor_movement(series_split);

  auto get_boundary_machine_view_assignments = [&](std::unordered_set<parallel_layer_guid_t> const &layers) 
    -> std::unordered_set<MachineMapping>
  {
    std::unordered_map<parallel_layer_guid_t, std::unordered_set<MachineView>>
      allowed = generate_map(layers,
                             [&](parallel_layer_guid_t const &l) { 
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
        restrict_domain(partial_solution, get_leaves(get_pre_child(series_split))),
        assigned_pre_machine_views);

    MachineMappingResultTree pre_result = ({
      std::optional<MachineMappingResultTree> returned
        = get_optimal_machine_mapping_internal(result_cache, 
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
          restrict_domain(partial_solution, get_leaves(get_post_child(series_split))),
          assigned_post_machine_views);

      MachineMappingResultTree post_result = ({
        std::optional<MachineMappingResultTree> returned 
          = get_optimal_machine_mapping_internal(result_cache, 
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

      result = minimize_cost(result, make_series_split(cost_across_split, pre_result, post_result));
    }
  }

  return result;
}



MachineMappingResultTree get_optimal_machine_mapping_internal(
    MachineMappingCache &result_cache,                                                          
    MachineMappingContext const &context,
    MMProblemTreeParallelSplit const &parallel,
    MachineSpecification const &resources,
    MachineMappingConstraints const &partial_solution) {

  MachineMappingResult optimal_result = [&] {
    MMProblemTreeSeriesSplit series = MMProblemTreeSeriesSplit{
      MMProblemTreeSeriesSplitLabel{empty_abstracted_tensor_set_movement()},
      parallel.left,
      parallel.right,
    };
        
    return get_optimal_machine_mapping_internal(result_cache,
                                                context,
                                                series,
                                                resources,
                                                partial_solution);
  }();

  MachineMappingConstraints left_sub_solution = restrict_domain(partial_solution,
                                                            get_leaves(parallel.left));
  MachineMappingConstraints right_sub_solution = restrict_domain(partial_solution,
                                                             get_leaves(parallel.right));

  for (auto const &resource_split : get_machine_resource_splits(resources)) {
    MachineMappingResult left_result = 
      get_optimal_machine_mapping_internal(result_cache,
                                           context,
                                           parallel.left,
                                           resource_split.first,
                                           left_sub_solution);
    MachineMappingResult right_result = 
      get_optimal_machine_mapping_internal(result_cache,
                                           context,
                                           parallel.right,
                                           resource_split.second,
                                           right_sub_solution);

    minimize_runtime(
        optimal_result,
        parallel_combine(left_result, right_result));
  }

  return optimal_result;
}

MachineMappingResultTree get_optimal_machine_mapping_internal(
    MachineMappingCache &result_cache,
    MachineMappingContext const &context,
    PCGOperatorAttrs const &layer,
    MachineSpecification const &resource,
    MachineMappingConstraints const &constraints) {

  assert (get_all_layers(constraints, IncludeUnconstrained{true}) == std::unordered_set{layer});

  MachineMapping concrete_mapping = require_fully_constrained(constraints);

  float cost = estimate_layer_cost(context.transitive_reduced_pcg.full_pcg, 
                                   context.cost_estimator,
                                   layer, 
                                   concrete_mapping.machine_views.at(layer));

  return make_leaf_node(
    /*runtime=*/cost,
    /*machine_mapping=*/concrete_mapping,
  };
}

} // namespace FlexFlow
