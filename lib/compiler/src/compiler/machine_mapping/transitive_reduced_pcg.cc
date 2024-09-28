#include "compiler/machine_mapping/transitive_reduced_pcg.h"
#include "compiler/series_parallel/pcg_binary_series_split.h"
#include "compiler/series_parallel/pcg_binary_sp_decomposition.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.h"
#include "utils/containers/flatmap.h"
#include "utils/graph/dataflow_graph/algorithms/transitive_reduced_dataflow_graph/get_transitive_reduced_outputs_across_split.h"
#include "utils/graph/digraph/algorithms/get_edges_from_subgraph_to_subgraph.h"
#include "utils/graph/digraph/algorithms/get_predecessors.h"
#include "utils/graph/digraph/algorithms/get_successors.h"
#include "utils/graph/digraph/algorithms/transitive_reduction.h"
#include "utils/graph/dataflow_graph/algorithms/transitive_reduced_dataflow_graph/get_transitive_reduced_edges_across_split.h"
#include "utils/graph/dataflow_graph/algorithms/transitive_reduced_dataflow_graph/get_transitive_reduced_boundary_nodes_for_split.h"

namespace FlexFlow {

TransitiveReducedDataflowGraphView get_underlying_transitive_reduced_dataflow_graph(TransitiveReducedPCG const &tr_pcg) {
  return TransitiveReducedDataflowGraphView{
    /*full_dataflow_graph=*/tr_pcg.full_pcg.raw_graph,
    /*transitive_reduction=*/tr_pcg.transitive_reduction,
  };
}

TransitiveReducedPCG pcg_get_transitive_reduction(ParallelComputationGraph const &pcg) {
  DiGraphView raw_digraph = pcg.raw_graph;
  DiGraphView transitive_reduced = transitive_reduction(raw_digraph);

  return TransitiveReducedPCG{
    /*pcg=*/pcg,
    /*transitive_reduction=*/transitive_reduced,
  };
}

std::unordered_set<ParallelComputationGraphEdge> 
  pcg_get_transitive_reduced_edges_across_split(TransitiveReducedPCG const &tr_pcg,
                                            PCGBinarySeriesSplit const &split) {

  TransitiveReducedDataflowGraphView raw_tr_g = get_underlying_transitive_reduced_dataflow_graph(tr_pcg);

  BinarySeriesSplit raw_split = get_raw_graph_series_split(split);

  std::unordered_set<DataflowEdge> raw_edges = get_transitive_reduced_edges_across_split(raw_tr_g, raw_split);

  return transform(raw_edges,
                   [](DataflowEdge const &e) { return ParallelComputationGraphEdge{e}; });
}

std::unordered_set<parallel_tensor_guid_t> 
  pcg_get_transitive_reduced_tensors_across_split(TransitiveReducedPCG const &tr_pcg,
                                              PCGBinarySeriesSplit const &split) {
  TransitiveReducedDataflowGraphView raw_tr_g = get_underlying_transitive_reduced_dataflow_graph(tr_pcg);

  BinarySeriesSplit raw_split = get_raw_graph_series_split(split);

  std::unordered_set<DataflowOutput> raw_outputs = get_transitive_reduced_outputs_across_split(raw_tr_g, raw_split);

  return transform(raw_outputs,
                   [](DataflowOutput const &o) { return parallel_tensor_guid_t{o}; });
}

PCGSplitBoundaryLayers pcg_get_transitive_reduced_boundary_layers_for_split(TransitiveReducedPCG const &tr_pcg,
                                                                            PCGBinarySeriesSplit const &split) {
  TransitiveReducedDataflowGraphView raw_tr_g = get_underlying_transitive_reduced_dataflow_graph(tr_pcg);

  BinarySeriesSplit raw_split = get_raw_graph_series_split(split);

  SplitBoundaryNodes raw_boundary = get_transitive_reduced_boundary_nodes_for_split(raw_tr_g, raw_split);

  return PCGSplitBoundaryLayers{
    /*pre_split_boundary=*/transform(raw_boundary.pre_split_boundary, [](Node const &n) { return parallel_layer_guid_t{n}; }),
    /*post_split_boundary=*/transform(raw_boundary.post_split_boundary, [](Node const &n) { return parallel_layer_guid_t{n}; }),
  };
}


} // namespace FlexFlow
