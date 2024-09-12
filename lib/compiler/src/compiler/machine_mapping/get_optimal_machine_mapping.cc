#include "compiler/machine_mapping/get_optimal_machine_mapping.h"
#include "compiler/cost_estimator.h"
#include "compiler/machine_mapping/get_allowed_machine_views_list.h"
#include "compiler/machine_mapping/machine_mapping_result.h"
#include "compiler/machine_mapping/split_sp_decomposition.h"
#include "pcg/machine_specification.dtg.h"
#include "pcg/machine_specification.h"
#include "pcg/machine_view.dtg.h"
#include "pcg/machine_view.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "substitutions/sub_parallel_computation_graph.h"
#include "utils/containers.h"
#include "utils/containers/as_vector.h"
#include "utils/containers/contains.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/filter.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/get_only.h"
#include "utils/containers/keys.h"
#include "utils/containers/merge_maps.h"
#include "utils/containers/restrict_keys.h"
#include "utils/containers/set_minus.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/containers/values.h"
#include "utils/exception.h"
#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/graph_split.dtg.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/serial_parallel/get_serial_parallel_decomposition.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.h"
#include "utils/graph/serial_parallel/serial_parallel_splits.h"
#include "utils/overload.h"

namespace FlexFlow {

std::vector<std::pair<MachineSpecification, MachineSpecification>>
    get_resource_split(MachineSpecification const &resource) {
  std::vector<std::pair<MachineSpecification, MachineSpecification>> result;
  for (int i = 1; i < resource.num_nodes; ++i) {
    MachineSpecification sub_resource1 = resource, sub_resource2 = resource;
    sub_resource1.num_nodes = i;
    sub_resource2.num_nodes = resource.num_nodes - i;
    result.push_back(std::make_pair(sub_resource1, sub_resource2));
  }
  return result;
}

GraphSplit
    get_graph_split(SerialParallelDecomposition const &pre_decomposition,
                    SerialParallelDecomposition const &post_decomposition) {
  return GraphSplit{get_nodes(pre_decomposition),
                    get_nodes(post_decomposition)};
}

float singleton_subgraph_cost(
    ParallelComputationGraph const &pcg,
    CostEstimator const &estimator,
    parallel_layer_guid_t const &layer,
    std::unordered_map<parallel_tensor_guid_t, MachineView> const
        &machine_views) {
  // TODO: Replace it with the actual implementation.
  auto get_input_shapes = [&](parallel_layer_guid_t) {
    return std::vector<ParallelTensorShape>{};
  };
  auto get_weight_attrs = [&](parallel_layer_guid_t) {
    return std::vector<ParallelTensorAttrs>{};
  };
  auto get_output_attrss = [&](parallel_layer_guid_t) {
    return std::vector<ParallelTensorAttrs>{};
  };

  assert(contains_key(machine_views, get_layer_outputs(pcg, layer)[0]));
  MachineView layer_machine_view =
      machine_views.at(get_layer_outputs(pcg, layer)[0]);
  float computation_cost =
      estimator.estimate_cost(get_parallel_layer_attrs(pcg, layer).op_attrs,
                              get_input_shapes(layer),
                              get_weight_attrs(layer),
                              get_output_attrss(layer),
                              layer_machine_view);
  float communication_cost = 0;
  for (parallel_tensor_guid_t const &input : get_layer_inputs(pcg, layer)) {
    assert(contains_key(machine_views, input));
    communication_cost = std::max(
        communication_cost,
        estimator.estimate_cost(get_parallel_tensor_attrs(pcg, input).shape,
                                machine_views.at(input),
                                layer_machine_view));
  }
  return computation_cost + communication_cost;
}

MachineMappingResult get_optimal_machine_mapping(
    ParallelComputationGraph const &pcg,
    std::function<std::unordered_set<MachineView>(
        ParallelLayerAttrs const &, MachineSpecification const &)> const
        &allowed_machine_views,
    CostEstimator const &cost_estimator,
    MachineSpecification const &resources,
    MachineMappingCache &cached_subgraph_results) {

  MachineMappingContext context(
      pcg, cost_estimator, allowed_machine_views, cached_subgraph_results);
  MachineMappingResult result =
      get_optimal_machine_mapping_internal(context, resources);
  cached_subgraph_results = context.cached_subgraph_results;
  return result;
}

MachineMappingResult get_optimal_machine_mapping_internal(
    MachineMappingContext &context, MachineSpecification const &resources) {
  std::optional<SerialParallelDecomposition> decompn_optional =
      get_serial_parallel_decomposition(context.pcg.raw_graph);

  if (!decompn_optional) {
    throw mk_runtime_error("Failed to get serial parallel decomposition");
  }

  SerialParallelDecomposition decompn = decompn_optional.value();

  return get_optimal_machine_mapping_internal(context, decompn, resources, {});
}

MachineMappingResult get_optimal_machine_mapping_internal(
    MachineMappingContext &context,
    SerialParallelDecomposition const &decompn,
    MachineSpecification const &resource,
    std::unordered_map<parallel_tensor_guid_t, MachineView> const
        &fixed_machine_views) {

  MachineMappingState state(decompn, resource, fixed_machine_views);
  std::optional<MachineMappingResult> cached_result =
      context.cached_subgraph_results.load(state);
  if (cached_result) {
    return cached_result.value();
  }

  MachineMappingResult result = decompn.visit<MachineMappingResult>(
      overload{[&](SerialSplit const &serial) {
                 return get_optimal_machine_mapping_internal(
                     context, serial, resource, fixed_machine_views);
               },
               [&](ParallelSplit const &parallel) {
                 return get_optimal_machine_mapping_internal(
                     context, parallel, resource, fixed_machine_views);
               },
               [&](Node const &node) {
                 return get_optimal_machine_mapping_internal(
                     context, node, resource, fixed_machine_views);
               }});

  context.cached_subgraph_results.save(state, result);
  return result;
}

MachineMappingResult get_optimal_machine_mapping_internal(
    MachineMappingContext &context,
    SerialSplit const &serial,
    MachineSpecification const &resource,
    std::unordered_map<parallel_tensor_guid_t, MachineView> const
        &fixed_machine_views) {
  MachineMappingResult optimal_result = get_infinity_machine_mapping_result();

  auto [decompn1, decompn2] = split_sp_decomposition(serial);

  GraphSplit graph_split = get_graph_split(decompn1, decompn2);

  auto is_subgraph_input = [&](std::unordered_set<Node> const &subgraph_nodes,
                               parallel_tensor_guid_t const &input_tensor) {
    return !contains(subgraph_nodes, input_tensor.raw_graph_output.node);
  };

  std::unordered_set<parallel_tensor_guid_t> all_edges1 =
      set_union(transform(graph_split.first, [&](Node const &node) {
        return unordered_set_of(
            get_layer_outputs(context.pcg, parallel_layer_guid_t(node)));
      }));
  std::unordered_set<parallel_tensor_guid_t> all_edges2 =
      set_union(transform(graph_split.second, [&](Node const &node) {
        return unordered_set_of(
            get_layer_inputs(context.pcg, parallel_layer_guid_t(node)));
      }));
  std::unordered_set<parallel_tensor_guid_t> split_edges =
      filter(all_edges2, [&](parallel_tensor_guid_t const &input_tensor) {
        return is_subgraph_input(graph_split.second, input_tensor);
      });

  std::unordered_map<parallel_tensor_guid_t, MachineView> fixed_machine_views1 =
      restrict_keys(fixed_machine_views, all_edges1);
  std::unordered_map<parallel_tensor_guid_t, MachineView> fixed_machine_views2 =
      restrict_keys(fixed_machine_views, all_edges2);
  std::vector<std::unordered_map<parallel_tensor_guid_t, MachineView>>
      machine_views_list_for_split_edges =
          get_allowed_src_machine_views_list(context, split_edges, resource);

  for (std::unordered_map<parallel_tensor_guid_t, MachineView> const
           &machine_views_for_split_edge : machine_views_list_for_split_edges) {
    minimize_runtime(
        optimal_result,
        sequential_combine(
            get_optimal_machine_mapping_internal(
                context,
                decompn1,
                resource,
                merge_maps(fixed_machine_views1, machine_views_for_split_edge)),
            get_optimal_machine_mapping_internal(
                context,
                decompn2,
                resource,
                merge_maps(fixed_machine_views2,
                           machine_views_for_split_edge))));
  }

  return optimal_result;
}

MachineMappingResult get_optimal_machine_mapping_internal(
    MachineMappingContext &context,
    ParallelSplit const &parallel,
    MachineSpecification const &resource,
    std::unordered_map<parallel_tensor_guid_t, MachineView> const
        &fixed_machine_views) {
  auto [decompn1, decompn2] = split_sp_decomposition(parallel);

  GraphSplit graph_split = get_graph_split(decompn1, decompn2);

  std::unordered_set<parallel_tensor_guid_t> all_edges1 =
      set_union(transform(graph_split.first, [&](Node const &node) {
        return unordered_set_of(
            get_layer_outputs(context.pcg, parallel_layer_guid_t(node)));
      }));
  std::unordered_set<parallel_tensor_guid_t> all_edges2 =
      set_union(transform(graph_split.second, [&](Node const &node) {
        return unordered_set_of(
            get_layer_inputs(context.pcg, parallel_layer_guid_t(node)));
      }));
  std::unordered_map<parallel_tensor_guid_t, MachineView> fixed_machine_views1 =
      restrict_keys(fixed_machine_views, all_edges1);
  std::unordered_map<parallel_tensor_guid_t, MachineView> fixed_machine_views2 =
      restrict_keys(fixed_machine_views, all_edges2);

  MachineMappingResult optimal_result = sequential_combine(
      get_optimal_machine_mapping_internal(
          context, decompn1, resource, fixed_machine_views1),
      get_optimal_machine_mapping_internal(
          context, decompn2, resource, fixed_machine_views2));

  for (auto const &resource_split : get_resource_split(resource)) {
    minimize_runtime(
        optimal_result,
        parallel_combine(
            get_optimal_machine_mapping_internal(
                context, decompn1, resource_split.first, fixed_machine_views1),
            get_optimal_machine_mapping_internal(context,
                                                 decompn2,
                                                 resource_split.second,
                                                 fixed_machine_views2)));
  }

  return optimal_result;
}

MachineMappingResult get_optimal_machine_mapping_internal(
    MachineMappingContext &context,
    Node const &node,
    MachineSpecification const &resource,
    std::unordered_map<parallel_tensor_guid_t, MachineView> const
        &fixed_machine_views) {

  parallel_layer_guid_t layer = parallel_layer_guid_t(node);
  std::unordered_set<parallel_tensor_guid_t> machine_views_not_fixed =
      set_minus(unordered_set_of(get_layer_outputs(context.pcg, layer)),
                keys(fixed_machine_views));

  std::vector<std::unordered_map<parallel_tensor_guid_t, MachineView>>
      machine_views_list_for_not_fixed = get_allowed_src_machine_views_list(
          context, machine_views_not_fixed, resource);

  MachineMappingResult optimal_result = get_infinity_machine_mapping_result();

  for (std::unordered_map<parallel_tensor_guid_t, MachineView> const
           &machine_views_for_not_fixed : machine_views_list_for_not_fixed) {
    std::unordered_map<parallel_tensor_guid_t, MachineView> full_machine_views =
        merge_maps(fixed_machine_views, machine_views_for_not_fixed);
    float runtime = singleton_subgraph_cost(
        context.pcg, context.cost_estimator, layer, full_machine_views);
    MachineMapping machine_mapping =
        MachineMapping{std::unordered_map<parallel_layer_guid_t, MachineView>{
            {layer,
             full_machine_views.at(get_layer_outputs(context.pcg, layer)[0])},
        }};
    MachineMappingResult curr_result =
        MachineMappingResult(runtime, machine_mapping);
    minimize_runtime(optimal_result, curr_result);
  }

  return optimal_result;
}

} // namespace FlexFlow
