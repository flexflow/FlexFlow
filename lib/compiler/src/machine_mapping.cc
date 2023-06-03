#include "compiler/machine_mapping.h"
#include "graph_utils.h"

namespace FlexFlow {

MachineMapping::MachineMapping(float runtime,
                   std::unordered_map<Node, MachineView> machine_views)
    : runtime(runtime), machine_views(machine_views) {}

MachineMapping MachineMapping::sequential_combine(MachineMapping const &s1, MachineMapping const &s2) {
  return MachineMapping(s1.runtime + s2.runtime,
                  merge_maps(s1.machine_views, s2.machine_views));
}

MachineMapping MachineMapping::parallel_combine(MachineMapping const &s1, MachineMapping const &s2) {
  return MachineMapping(std::max(s1.runtime, s2.runtime),
                  merge_maps(s1.machine_views, s2.machine_views));
}

MachineMapping MachineMapping::infinity() {
  return MachineMapping(std::numeric_limits<float>::infinity(),
                  std::unordered_map<Node, MachineView>{});
}

bool MachineMappingRuntimeCmp::operator()(MachineMapping const &lhs, MachineMapping const &rhs) {
  return lhs.runtime < rhs.runtime;
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
template <typename T>
std::pair<SerialParallelDecomposition, SerialParallelDecomposition>
    decompose(T const &t) {
  if (t.children.size() == 2) {
    return {widen<SerialParallelDecomposition>(t.children[0]),
            widen<SerialParallelDecomposition>(t.children[1])};
  }
  T decompn1 = t;
  decompn1.children.pop_back();
  return {decompn1, widen<SerialParallelDecomposition>(t.children.back())};
}

GraphSplit
    get_graph_split(SerialParallelDecomposition const &pre_decomposition,
                    SerialParallelDecomposition const &post_decomposition) {
  return {get_nodes(pre_decomposition), get_nodes(post_decomposition)};
}

std::pair<SubParallelComputationGraph, SubParallelComputationGraph>
    apply_split(SubParallelComputationGraph const &g, GraphSplit const &split) {
  OpenMultiDiGraphView g1 =
      unsafe_view_as_subgraph(OpenMultiDiGraph(g), split.first);
  OpenMultiDiGraphView g2 =
      unsafe_view_as_subgraph(OpenMultiDiGraph(g), split.second);

  if (get_cut(OpenMultiDiGraph(g), split).size() > 0) {
    // Sequential split
    if (get_open_sinks(g1).size() <= get_open_sources(g2).size()) {
      // get_open_sinks(*g1).size() should be 1 in perfect sp graphs
      return {get_subgraph(g,
                           split.first,
                           InputSettings::INCLUDE_INPUTS,
                           OutputSettings::EXCLUDE_OUTPUTS),
              get_subgraph(g,
                           split.second,
                           InputSettings::INCLUDE_INPUTS,
                           OutputSettings::INCLUDE_OUTPUTS)};
    } else {
      return {get_subgraph(g,
                           split.first,
                           InputSettings::INCLUDE_INPUTS,
                           OutputSettings::INCLUDE_OUTPUTS),
              get_subgraph(g,
                           split.second,
                           InputSettings::EXCLUDE_INPUTS,
                           OutputSettings::INCLUDE_OUTPUTS)};
    }
  } else {
    // Parallel split
    return {get_subgraph(g,
                         split.first,
                         InputSettings::INCLUDE_INPUTS,
                         OutputSettings::INCLUDE_OUTPUTS),
            get_subgraph(g,
                         split.second,
                         InputSettings::INCLUDE_INPUTS,
                         OutputSettings::INCLUDE_OUTPUTS)};
  }
}

float estimate_cost(
    SubParallelComputationGraph const &g,
    ICostEstimator const &estimator,
    std::unordered_map<Node, MachineView> const &device_mapping) {}

struct OptimalCost {
  OptimalCost(
      SubParallelComputationGraph const &g,
      ICostEstimator const &cost_estimator,
      MachineSpecification const &resource,
      optional<MachineView> const &source_machine_view, // assume perfect SP
      optional<MachineView> const &sink_machine_view,
      std::function<std::unordered_set<MachineView>(
          PCGOperatorAttrs const &, MachineSpecification const &)> const
          &allowed_machine_views,
      std::unordered_map<size_t, MachineMapping> &cached_subgraph_costs)
      : g(g), cost_estimator(cost_estimator), resource(resource),
        source_machine_view(source_machine_view),
        sink_machine_view(sink_machine_view),
        allowed_machine_views(allowed_machine_views),
        cached_subgraph_costs(cached_subgraph_costs) {}

  // TODO: move them out of the functor
  template <typename T>
  size_t hash_state(T const &sp_decomposition) const {
    size_t h = std::hash<T>{}(sp_decomposition);
    hash_combine(h, resource);
    hash_combine(h, source_machine_view);
    hash_combine(h, sink_machine_view);
    return h;
  }

  optional<MachineMapping> load_result_from_cache(size_t hash_value) const {
    if (contains_key(cached_subgraph_costs, hash_value)) {
      return make_optional(cached_subgraph_costs.at(hash_value));
    }
    return nullopt;
  }

  void save_result_to_cache(size_t hash_value, MachineMapping const &strategy) const {
    assert(!contains_key(cached_subgraph_costs, hash_value));
    cached_subgraph_costs.emplace(hash_value, strategy);
  }

  template <typename T>
  MachineMapping operator()(T const &t) const {
    size_t state_hash_value = hash_state(t);
    optional<MachineMapping> cached_result = load_result_from_cache(state_hash_value);
    if (cached_result) {
      return cached_result.value();
    }

    MachineMapping result = this->optimal_cost(t);

    save_result_to_cache(state_hash_value, result);
    return result;
  }

  MachineMapping optimal_cost(Serial const &serial) const {
    // return sum(vector_transform([&](variant<Parallel, Node> const &t) {
    // return visit(*this, t); }, serial.children));
    SerialParallelDecomposition pre_decompn, post_decompn;
    std::tie(pre_decompn, post_decompn) = decompose(serial);

    auto subgraphs = apply_split(g, get_graph_split(pre_decompn, post_decompn));
    SubParallelComputationGraph pre_graph = subgraphs.first,
                                post_graph = subgraphs.second;

    std::unordered_set<Node> pre_graph_sinks =
        get_closed_sinks(OpenMultiDiGraph(pre_graph));
    std::unordered_set<Node> post_graph_sources =
        get_closed_sources(OpenMultiDiGraph(post_graph));

    assert(pre_graph_sinks.size() + post_graph_sources.size() ==
           1); // assume perfect SP

    Node const &split_point =
        get_only(set_union(pre_graph_sinks, post_graph_sources));

    MachineMapping optimal_result = MachineMapping::infinity();

    for (MachineView const &mv :
         allowed_machine_views(g.at(split_point), resource)) {
      optional<MachineView> pre_sink_mv =
          contains(pre_graph_sinks, split_point) ? make_optional(mv) : nullopt;
      optional<MachineView> post_source_mv =
          contains(post_graph_sources, split_point) ? make_optional(mv)
                                                    : nullopt;
      minimize(
          optimal_result,
          MachineMapping::sequential_combine(visit(OptimalCost(pre_graph,
                                                         cost_estimator,
                                                         resource,
                                                         source_machine_view,
                                                         pre_sink_mv,
                                                         allowed_machine_views,
                                                         cached_subgraph_costs),
                                             pre_decompn),
                                       visit(OptimalCost(post_graph,
                                                         cost_estimator,
                                                         resource,
                                                         post_source_mv,
                                                         sink_machine_view,
                                                         allowed_machine_views,
                                                         cached_subgraph_costs),
                                             post_decompn)));
    }

    return optimal_result;
  }

  MachineMapping optimal_cost(Parallel const &parallel) const {
    SerialParallelDecomposition decompn1, decompn2;

    std::tie(decompn1, decompn2) = decompose(parallel);

    auto subgraphs = apply_split(g, get_graph_split(decompn1, decompn2));
    SubParallelComputationGraph g1 = subgraphs.first, g2 = subgraphs.second;

    MachineMapping optimal_result =
        MachineMapping::sequential_combine(visit(OptimalCost(g1,
                                                       cost_estimator,
                                                       resource,
                                                       source_machine_view,
                                                       sink_machine_view,
                                                       allowed_machine_views,
                                                       cached_subgraph_costs),
                                           decompn1),
                                     visit(OptimalCost(g2,
                                                       cost_estimator,
                                                       resource,
                                                       source_machine_view,
                                                       sink_machine_view,
                                                       allowed_machine_views,
                                                       cached_subgraph_costs),
                                           decompn2));

    for (auto const &resource_split : get_resource_split(resource)) {
      minimize(
          optimal_result,
          MachineMapping::parallel_combine(visit(OptimalCost(g1,
                                                       cost_estimator,
                                                       resource_split.first,
                                                       source_machine_view,
                                                       sink_machine_view,
                                                       allowed_machine_views,
                                                       cached_subgraph_costs),
                                           decompn1),
                                     visit(OptimalCost(g2,
                                                       cost_estimator,
                                                       resource_split.second,
                                                       source_machine_view,
                                                       sink_machine_view,
                                                       allowed_machine_views,
                                                       cached_subgraph_costs),
                                           decompn2)));
    }

    return optimal_result;
  }

  MachineMapping optimal_cost(Node const &node) const {
    if (source_machine_view) {
      assert(get_closed_sources(OpenMultiDiGraph(g)).empty());
      assert(contains(allowed_machine_views(g.at(node), resource),
                      source_machine_view.value()));
      std::unordered_map<Node, MachineView> mv_map{
          {node, source_machine_view.value()}};
      return MachineMapping(estimate_cost(g, cost_estimator, mv_map), mv_map);
    } else if (sink_machine_view) {
      assert(get_closed_sinks(OpenMultiDiGraph(g)).empty());
      assert(contains(allowed_machine_views(g.at(node), resource),
                      sink_machine_view.value()));
      std::unordered_map<Node, MachineView> mv_map{
          {node, sink_machine_view.value()}};
      return MachineMapping(estimate_cost(g, cost_estimator, mv_map), mv_map);
    } else {
      MachineMapping optimal_result = MachineMapping::infinity();
      for (auto mv : allowed_machine_views(g.at(node), resource)) {
        std::unordered_map<Node, MachineView> mv_map{{node, mv}};
        minimize(optimal_result,
                 MachineMapping(estimate_cost(g, cost_estimator, mv_map), mv_map));
      }
      return optimal_result;
    }
  }

  SubParallelComputationGraph const &g;
  ICostEstimator const &cost_estimator;
  MachineSpecification const &resource;
  optional<MachineView> const &source_machine_view;
  optional<MachineView> const &sink_machine_view;
  std::function<std::unordered_set<MachineView>(
      PCGOperatorAttrs const &, MachineSpecification const &)> const
      &allowed_machine_views;
  std::unordered_map<size_t, MachineMapping> &cached_subgraph_costs;
};

MachineMapping optimal_cost(
    OptimizerPCG const &g,
    std::function<std::unordered_set<MachineView>(
        PCGOperatorAttrs const &, MachineSpecification const &)> const
        &allowed_machine_views,
    ICostEstimator const &cost_estimator,
    MachineSpecification const &resources,
    std::unordered_map<size_t, MachineMapping> &cached_subgraph_costs) {
  return visit(OptimalCost(pcg_to_subpcg(g),
                           cost_estimator,
                           resources,
                           nullopt,
                           nullopt,
                           allowed_machine_views,
                           cached_subgraph_costs),
               get_serial_parallel_decomposition(g));
}

}