#include "compiler/machine_mapping/transitive_reduced_pcg.h"
#include "compiler/series_parallel/pcg_binary_series_split.h"
#include "compiler/series_parallel/pcg_binary_sp_decomposition.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.h"
#include "utils/containers/flatmap.h"
#include "utils/graph/digraph/algorithms/get_edges_from_subgraph_to_subgraph.h"
#include "utils/graph/digraph/algorithms/get_predecessors.h"
#include "utils/graph/digraph/algorithms/get_successors.h"
#include "utils/graph/digraph/algorithms/transitive_reduction.h"

namespace FlexFlow {

TransitiveReducedPCG get_pcg_transitive_reduction(ParallelComputationGraph const &pcg) {
  DiGraphView raw_digraph = pcg.raw_graph;
  DiGraphView transitively_reduced = transitive_reduction(raw_digraph);

  return TransitiveReducedPCG{
    /*pcg=*/pcg,
    /*transitive_reduction=*/transitively_reduced,
  };
}

std::unordered_set<parallel_layer_guid_t> get_transitive_reduced_predecessors(TransitiveReducedPCG const &tr_pcg,
                                                                                parallel_layer_guid_t const &layer) {
  std::unordered_set<Node> raw_predecessors = get_predecessors(tr_pcg.transitive_reduction, layer.raw_graph_node); 
  return transform(raw_predecessors, [](Node const &n) { return parallel_layer_guid_t{n}; });
}

std::unordered_set<parallel_layer_guid_t> get_transitive_reduced_successors(TransitiveReducedPCG const &tr_pcg,
                                                                            parallel_layer_guid_t const &layer) {
  std::unordered_set<Node> raw_successors = get_successors(tr_pcg.transitive_reduction, layer.raw_graph_node);
  return transform(raw_successors, [](Node const &n) { return parallel_layer_guid_t{n}; });
}

std::unordered_set<ParallelComputationGraphEdge> 
  get_transitively_reduced_edges_across_split(TransitiveReducedPCG const &tr_pcg,
                                              PCGBinarySeriesSplit const &split) {
  std::unordered_set<parallel_layer_guid_t> src_subgraph = unordered_set_of(get_parallel_layers(get_left_child(split)));
  std::unordered_set<parallel_layer_guid_t> dst_subgraph = unordered_set_of(get_parallel_layers(get_right_child(split)));

  std::unordered_set<Node> raw_src_subgraph = transform(src_subgraph, [](parallel_layer_guid_t const &l) { return l.raw_graph_node; });
  std::unordered_set<Node> raw_dst_subgraph = transform(dst_subgraph, [](parallel_layer_guid_t const &l) { return l.raw_graph_node; });

  std::unordered_set<DirectedEdge> raw_edges = get_edges_from_subgraph_to_subgraph(tr_pcg.transitive_reduction, 
                                                                                   raw_src_subgraph,
                                                                                   raw_dst_subgraph);

  return flatmap(raw_edges, 
                 [&](DirectedEdge const &e) {
                   return get_pcg_edges_from_layer_to_layer(tr_pcg.full_pcg, 
                                                            parallel_layer_guid_t{e.src}, 
                                                            parallel_layer_guid_t{e.dst});
                 });
}

std::unordered_set<parallel_tensor_guid_t> 
  get_transitively_reduced_tensors_across_split(TransitiveReducedPCG const &tr_pcg,
                                                PCGBinarySeriesSplit const &split) {
  return transform(get_transitively_reduced_edges_across_split(tr_pcg, split),
                   [](ParallelComputationGraphEdge const &e) { return get_parallel_tensor(e); });
}

std::pair<
  std::unordered_set<parallel_layer_guid_t>,
  std::unordered_set<parallel_layer_guid_t>
> get_split_transitively_reduced_boundary_layers(TransitiveReducedPCG const &tr_pcg,
                                                 PCGBinarySeriesSplit const &split) {
  std::unordered_set<ParallelComputationGraphEdge> edges = get_transitive_reduced_edges_across_split(tr_pcg, split);

  std::unordered_set<parallel_layer_guid_t> src_boundary_layers = transform(edges,
                                                                            [](ParallelComputationGraphEdge const &e) { return get_src_layer(e); });

  std::unordered_set<parallel_layer_guid_t> dst_boundary_layers = transform(edges,
                                                                            [](ParallelComputationGraphEdge const &e) { return get_dst_layer(e); });

  return {
    src_boundary_layers,
    dst_boundary_layers,
  };
}


} // namespace FlexFlow
