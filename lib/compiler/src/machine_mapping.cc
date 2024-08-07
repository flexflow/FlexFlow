#include "compiler/machine_mapping.h"
#include "compiler/cost_estimate.h"
#include "compiler/graph_utils.h"
#include "pcg/machine_specification.dtg.h"
#include "pcg/machine_specification.h"
#include "pcg/machine_view.dtg.h"
#include "pcg/machine_view.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "utils/containers.h"
#include "utils/containers/are_disjoint.h"
#include "utils/containers/as_vector.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/get_only.h"
#include "utils/containers/keys.h"
#include "utils/exception.h"
#include "utils/graph/graph_split.dtg.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/open_dataflow_graph/algorithms.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_subgraph.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.h"
#include "utils/graph/serial_parallel/serial_parallel_splits.h"

namespace FlexFlow {

MachineMapping combine(MachineMapping const &s1, MachineMapping const &s2) {
  return MachineMapping{merge_maps(s1.machine_views, s2.machine_views)};
}

bool nodes_are_disjoint(MachineMapping const &m1, MachineMapping const &m2) {
  return are_disjoint(keys(m1.machine_views), keys(m2.machine_views));
}

OptimalCostResult
    OptimalCostResult::sequential_combine(OptimalCostResult const &s1,
                                          OptimalCostResult const &s2) {
  return OptimalCostResult{s1.runtime + s2.runtime,
                           combine(s1.machine_mapping, s2.machine_mapping)};
}

OptimalCostResult
    OptimalCostResult::parallel_combine(OptimalCostResult const &s1,
                                        OptimalCostResult const &s2) {
  return OptimalCostResult{std::max(s1.runtime, s2.runtime),
                           combine(s1.machine_mapping, s2.machine_mapping)};
}

OptimalCostResult OptimalCostResult::infinity() {
  return {std::numeric_limits<float>::infinity(),
          MachineMapping{std::unordered_map<Node, MachineView>{}}};
}

bool OptimalCostRuntimeCmp::operator()(OptimalCostResult const &lhs,
                                       OptimalCostResult const &rhs) {
  return lhs.runtime < rhs.runtime;
}

std::optional<OptimalCostResult>
    OptimalCostCache::load(OptimalCostState const &state) const {
  if (contains_key(cache, state)) {
    OptimalCostResult result = cache.at(state);
    return std::make_optional(result);
  }
  return std::nullopt;
}

void OptimalCostCache::save(OptimalCostState const &state,
                            OptimalCostResult const &result) {
  assert(!contains_key(cache, state));
  cache.emplace(state, result);
}

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

// We may replace this by having unflattened AST
std::pair<SerialParallelDecomposition, SerialParallelDecomposition>
    decompose(SerialSplit const &serial) {
  if (serial.children.size() == 2) {
    return {widen<SerialParallelDecomposition>(serial.children[0]),
            widen<SerialParallelDecomposition>(serial.children[1])};
  }
  SerialSplit decompn1 = serial;
  decompn1.children.pop_back();
  return {SerialParallelDecomposition(decompn1),
          widen<SerialParallelDecomposition>(serial.children.back())};
}

std::pair<SerialParallelDecomposition, SerialParallelDecomposition>
    decompose(ParallelSplit const &parallel) {
  if (parallel.children.size() == 2) {
    std::vector<SerialParallelDecomposition> children =
        transform(as_vector(parallel.children), [&](auto const &child) {
          return widen<SerialParallelDecomposition>(child);
        });
    return {children[0], children[1]};
  }
  ParallelSplit decompn1 = parallel;
  std::variant<SerialSplit, Node> child = *parallel.children.begin();
  decompn1.children.erase(child);
  return {SerialParallelDecomposition(decompn1),
          widen<SerialParallelDecomposition>(child)};
}

GraphSplit
    get_graph_split(SerialParallelDecomposition const &pre_decomposition,
                    SerialParallelDecomposition const &post_decomposition) {
  return GraphSplit{get_nodes(pre_decomposition),
                    get_nodes(post_decomposition)};
}

float estimate_cost(SubParallelComputationGraph const &g,
                    CostEstimator const &estimator,
                    MachineMapping const &device_mapping,
                    std::unordered_map<OpenDataflowEdge, MachineView> const
                        &frontier_machine_views) {
  // TODO: Consider parallelism
  float cost = 0;
  // for (Node const &node : get_nodes(g.raw_graph)) {
  //   std::vector<OpenDataflowEdge> incoming_edges =
  //       get_incoming_edges(g.raw_graph, node);
  //   std::vector<ParallelTensorShape> inputs =
  //       transform(incoming_edges,
  //                 [&](OpenDataflowEdge const &input_edge) {
  //                   return g.raw_graph.at(input_edge).get_shape();
  //                 });
  //   cost += estimator.estimate_cost(
  //       g.raw_graph.at(node).op_attrs, inputs,
  //       device_mapping.machine_views.at(node));
  // }
  return cost;
}

void minimize_runtime(OptimalCostResult &m1, OptimalCostResult const &m2) {
  minimize(m1, m2, OptimalCostRuntimeCmp{});
}

struct MachineMappingSearcher {
  MachineMappingSearcher(
      CostEstimator cost_estimator,
      std::function<std::unordered_set<MachineView>(
          ParallelLayerAttrs const &, MachineSpecification const &)> const
          &allowed_machine_views,
      OptimalCostCache &cached_subgraph_costs)
      : cost_estimator(cost_estimator),
        allowed_machine_views(allowed_machine_views),
        cached_subgraph_costs(cached_subgraph_costs) {}

  CostEstimator cost_estimator;
  std::function<std::unordered_set<MachineView>(ParallelLayerAttrs const &,
                                                MachineSpecification const &)>
      allowed_machine_views;
  OptimalCostCache &cached_subgraph_costs;

  struct OptimalCostFunctor {
    OptimalCostFunctor(
        MachineMappingSearcher *searcher,
        SubParallelComputationGraph const &g,
        MachineSpecification resource,
        std::unordered_map<Node, MachineView> given_machine_views,
        std::unordered_map<OpenDataflowEdge, MachineView>
            frontier_machine_views)
        : searcher(searcher), g(g), resource(resource),
          given_machine_views(given_machine_views),
          frontier_machine_views(frontier_machine_views) {}

    MachineMappingSearcher *searcher;
    SubParallelComputationGraph const &g;
    MachineSpecification resource;
    std::unordered_map<Node, MachineView> given_machine_views;
    std::unordered_map<OpenDataflowEdge, MachineView> frontier_machine_views;

    template <typename T>
    OptimalCostResult operator()(T const &t) {
      OptimalCostState state{SerialParallelDecomposition{t},
                             resource,
                             given_machine_views,
                             frontier_machine_views};
      std::optional<OptimalCostResult> cached_result =
          searcher->cached_subgraph_costs.load(state);

      if (cached_result) {
        return cached_result.value();
      }
      OptimalCostResult result = searcher->optimal_cost(
          t, g, resource, given_machine_views, frontier_machine_views);

      searcher->cached_subgraph_costs.save(state, result);
      return result;
    }
  };

  OptimalCostResult
      optimal_cost(SubParallelComputationGraph const &g,
                   MachineSpecification resource,
                   SerialParallelDecomposition const &sp_decomposition) {
    return std::visit(OptimalCostFunctor(this, g, resource, {}, {}),
                      sp_decomposition.raw_variant);
  }

  OptimalCostResult optimal_cost(
      SerialSplit const &serial,
      SubParallelComputationGraph const &g,
      MachineSpecification const &resource,
      std::unordered_map<Node, MachineView> const &given_machine_views,
      std::unordered_map<OpenDataflowEdge, MachineView> const
          &frontier_machine_views) {
    NOT_IMPLEMENTED();
    // OptimalCostResult optimal_result = OptimalCostResult::infinity();

    // auto decomposed = decompose(serial);
    // SerialParallelDecomposition pre_decompn = decomposed.first;
    // SerialParallelDecomposition post_decompn = decomposed.second;

    // GraphSplit graph_split = get_graph_split(pre_decompn, post_decompn);
    // SubParallelComputationGraph pre_graph =
    //     get_subgraph<OpenMultiDiSubgraphView>(g, graph_split.first);
    // SubParallelComputationGraph post_graph =
    //     get_subgraph<DownwardOpenMultiDiSubgraphView>(g, graph_split.second);

    // std::unordered_set<Node> post_graph_sources =
    //     get_closed_sources(post_graph);

    // assert(post_graph_sources.size() == 1); // assume perfect SP

    // Node split_point = get_only(post_graph_sources);
    // OutputMultiDiEdge split_edge = get_only(get_open_outputs(pre_graph));

    // for (MachineView const &mv :
    //      allowed_machine_views(g.raw_graph.at(split_point), resource)) {
    //   std::unordered_map<Node, MachineView> new_given_machine_views =
    //       given_machine_views;
    //   new_given_machine_views.emplace(split_point, mv);
    //   std::unordered_map<OpenDataflowEdge, MachineView>
    //       new_frontier_machine_views = frontier_machine_views;
    //   new_frontier_machine_views.emplace(split_edge, mv);
    //   minimize_runtime(
    //       optimal_result,
    //       OptimalCostResult::sequential_combine(
    //           std::visit(OptimalCostFunctor(this,
    //                                         pre_graph,
    //                                         resource,
    //                                         given_machine_views,
    //                                         new_frontier_machine_views),
    //                      pre_decompn.raw_variant),
    //           std::visit(OptimalCostFunctor(this,
    //                                         post_graph,
    //                                         resource,
    //                                         new_given_machine_views,
    //                                         frontier_machine_views),
    //                      post_decompn.raw_variant)));
    // }

    // return optimal_result;
  }

  OptimalCostResult optimal_cost(
      ParallelSplit const &parallel,
      SubParallelComputationGraph const &g,
      MachineSpecification const &resource,
      std::unordered_map<Node, MachineView> const &given_machine_views,
      std::unordered_map<OpenDataflowEdge, MachineView> const
          &frontier_machine_views) {

    NOT_IMPLEMENTED();
    // auto decomposed = decompose(parallel);
    // SerialParallelDecomposition decompn1 = decomposed.first;
    // SerialParallelDecomposition decompn2 = decomposed.second;

    // GraphSplit graph_split = get_graph_split(decompn1, decompn2);
    // SubParallelComputationGraph g1 = get_subgraph(g, graph_split.first),
    //                             g2 = get_subgraph(g, graph_split.second);

    // OptimalCostResult optimal_result = OptimalCostResult::sequential_combine(
    //     std::visit(OptimalCostFunctor(this,
    //                                   g1,
    //                                   resource,
    //                                   given_machine_views,
    //                                   frontier_machine_views),
    //                decompn1.raw_variant),
    //     std::visit(OptimalCostFunctor(this,
    //                                   g2,
    //                                   resource,
    //                                   given_machine_views,
    //                                   frontier_machine_views),
    //                decompn2.raw_variant));

    // for (auto const &resource_split : get_resource_split(resource)) {
    //   minimize_runtime(
    //       optimal_result,
    //       OptimalCostResult::parallel_combine(
    //           std::visit(OptimalCostFunctor(this,
    //                                         g1,
    //                                         resource_split.first,
    //                                         given_machine_views,
    //                                         frontier_machine_views),
    //                      decompn1.raw_variant),
    //           std::visit(OptimalCostFunctor(this,
    //                                         g2,
    //                                         resource_split.second,
    //                                         given_machine_views,
    //                                         frontier_machine_views),
    //                      decompn2.raw_variant)));
    // }

    // return optimal_result;
  }

  OptimalCostResult optimal_cost(
      Node const &node,
      SubParallelComputationGraph const &g,
      MachineSpecification const &resource,
      std::unordered_map<Node, MachineView> const &given_machine_views,
      std::unordered_map<OpenDataflowEdge, MachineView> const
          &frontier_machine_views) {
    if (contains_key(given_machine_views, node)) {
      assert(contains(allowed_machine_views(g.raw_graph.at(node), resource),
                      given_machine_views.at(node)));
      MachineMapping mv_map{given_machine_views};
      return {estimate_cost(g, cost_estimator, mv_map, frontier_machine_views),
              mv_map};
    } else {
      OptimalCostResult optimal_result = OptimalCostResult::infinity();
      for (auto mv : allowed_machine_views(g.raw_graph.at(node), resource)) {
        MachineMapping mv_map{{{node, mv}}};
        minimize_runtime(
            optimal_result,
            {estimate_cost(g, cost_estimator, mv_map, frontier_machine_views),
             mv_map});
      }
      return optimal_result;
    }
  }
};

OptimalCostResult optimal_cost(
    ParallelComputationGraph const &g,
    std::function<std::unordered_set<MachineView>(
        ParallelLayerAttrs const &, MachineSpecification const &)> const
        &allowed_machine_views,
    CostEstimator const &cost_estimator,
    MachineSpecification const &resources,
    OptimalCostCache &cached_subgraph_costs) {
  SerialParallelDecomposition sp_decomposition =
      get_serial_parallel_decomposition(g);
  SubParallelComputationGraph subpcg = pcg_to_subpcg(g);
  MachineMappingSearcher searcher(
      cost_estimator, allowed_machine_views, cached_subgraph_costs);
  return searcher.optimal_cost(subpcg, resources, sp_decomposition);
}

// bool is_valid_machine_view(MachineSpecification const &machinespec,
//  MachineView const &mv) {
// Note: we're checking the size of the machineview, not the last device id
// (e.g. consider 2D machine view of size 4x4 with stride 2 along each
// dimension, technically it could fit into a 3x3 machine spec but the machine
// view is really describing a 4x4 space)
// if (get_device_type(mv) == DeviceType::GPU) {
//   return get_size(mv) <=
//          machinespec.num_nodes * machinespec.num_gpus_per_node;
// } else if (get_device_type(mv) == DeviceType::CPU) {
//   return get_size(mv) <=
//          machinespec.num_nodes * machinespec.num_cpus_per_node;
// } else {
//   assert(false && "Unsupported DeviceType");
//   return false;
// }

// Add check that the last one doesn't go over
// Add other check that you can cram them into the 2D frame (there has to
// exist a bipartition of the dimension that fits)
// }

// bool is_valid_machine_view(MachineView const &mv,
//                            ParallelTensorShape const &shape) {
// std::unordered_set<size_t> unordered_mv_degrees =
//     without_order(get_point_dims(mv));
// std::unordered_set<size_t> unordered_tensor_degrees =
//     without_order(ff_ordered_shard_degrees(shape)) + {get_sum_degree(shape)}
//     + {get_discard_copy_degree(shape)}; // filter for the 1s (no parallelism)
// return unordered_mv_dims == unordered_tensor_dims;
// }

// WARNING: some machine_views returned are invalid, get allowed_machine_views
// for valid ones.
// static std::unordered_set<MachineView>
//     get_all_machine_views(MachineSpecification const &machinespec,
//                           ParallelTensorShape const &shape) {

// auto all_possible_strides =
//     [](std::vector<size_t> tensor_dims,
//        size_t total_devices,
//        size_t num_devices_used_by_tensor) {
//       size_t room_for_stride = total_devices / num_devices_used_by_tensor;
//       auto x = cartesian_product(replicate(range(1, room_for_stride + 1)),
//                                  tensor_dims.size());
//   return filter(x, product((elem-1 for elem in x)) <= room_for_stride);
//     }

// size_t total_devices = machinespec.num_nodes *
// machinespec.num_gpus_per_nodes; std::unordered_set<MachineView>
// machine_views; std::vector<size_t> tensor_dims; size_t
// num_devices_used_by_tensor = product(tensor_dims);
// std::unordered_set<std::unordered_set<size_t>> stride_sets =
//     make_stride_sets(tensor_dims, total_devices);
// for (std::vector<size_t> stride :
//      all_possible_strides(tensor_dims, total_devices)) {
//   for (int start_id = 0 :
//        start_id <= total_devices - num_devices_used_by_tensor + 1;
//        start_id++) {
//     std::vector<StridedRectangleSide> sides =
//         transform(zip(tensor_dims, stride));
//     MachineView mv = {start, StridedRectangle{sides}};
//     machine_views.insert(mv);
//   }
// }
// return machine_views;
// }

// static std::unordered_set<StartInvariantMachineView>
//     get_all_start_invariant_machine_views(
//         MachineSpecification const &machinespec,
//         ParallelTensorShape const &shape) {
//   NOT_IMPLEMENTED();
// }

// auto get_all_machine_views_to_tensor_dim_bijections(MachineView const &mv,
// ParallelTensorShape const &shape) {
//   NOT_IMPLEMENTED();
// }

// // do product (total num of devices vs total num of elements in the tensor).
// bool is_valid_machine_view(ParallelTensorShape const&
// output_shape, MachineView const& view) {
//   int is_dim = 0;
//   for (int i = 0; i < num_dims; i++) {
//     if (dims[i].parallel_idx != -1) {
//       is_dim++;
//       if (dims[i].parallel_idx > view.ndims) {
//         return false;
//       }
//       if (view.dim[dims[i].parallel_idx] != dims[i].degree) {
//         return false;
//       }
//     }
//   }
//   if (is_dim == 0) {
//     is_dim = 1;
//   }
//   if (is_dim != view.ndims) {
//     return false;
//   }
//   if (get_total_num_parts() != view.num_parts()) {
//     return false;
//   }
// return true;
// }

// std::unordered_set<MachineView>
//     get_allowed_machine_views(MachineSpecification const &machinespec,
//                               ParallelTensorShape const &shape) {
//   std::unordered_set<MachineView> operator_views =
//       get_all_machine_views(machinespec, shape);
//   // operator_views = filter(operator_views, [&](MachineView const& view)
//   // {return all_of(outputs(op), [&](ParallelTensorShape const&
//   output){return
//   // is_valid_machine_view(op, output, view);});
//   operator_views = filter(operator_views, [&](MachineView const &view) {
//     return is_valid_machine_view(shape, view);
//   });
//   operator_views = filter(operator_views, [&](MachineView const &view) {
//     return is_valid_machine_view(machinespec, view);
//   });
//   return operator_views;
// }

// // Ask the output shapes
// // Get the PCG

// std::vector<MachineView> SearchHelper::get_valid_machine_views(
//     Op const *op, MachineResource const &resource) const {
//   std::vector<MachineView> const cached_op_views;
//   std::vector<MachineView> valid_views;
//   for (size_t i = 0; i < this->model->all_valid_views.size(); i++) {
//     bool valid = true;
//     for (int j = 0; j < op->numOutputs; j++) {
//       if (!op->outputs[j]->is_valid_machine_view(
//               this->model->all_valid_views[i])) {
//         valid = false;
//         break;
//       }
//     }
//     if (valid) {
//       cached_op_views.push_back(this->model->all_valid_views[i]);
//     }
//   }

//   for (size_t i = 0; i < cached_op_views->size(); i++) {
//     if (resource.is_valid_machine_view(view)) {
//       valid_views.push_back(view);
//     }
//   }
//   return valid_views;
// }

} // namespace FlexFlow
