#include "utils/graph/serial_parallel/graph_generation.h"
#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/instances/unordered_set_dataflow_graph.h"
#include "utils/graph/node/algorithms.h"

namespace FlexFlow {

void parallel_extend_unsafe(DataflowGraph &g, DataflowGraphView const &ext) {
  for (Node const &node : get_nodes(ext)) {
    g.add_node_unsafe(node, get_inputs(ext, node), get_outputs(ext, node));
  }
}

void serial_extend_unsafe(DataflowGraph &g, DataflowGraphView const &ext) {
  // TODO(@lockshaw): This function signature is impossible to implement in
  // general, as there is no guarantee that the graph view ext actually has
  // source nodes with inputs Either the signature should be changed, or an
  // implementation should be added that throws an error if this problematic
  // case is found

  NOT_IMPLEMENTED();
}

DataflowGraph serial_composition(DataflowGraphView const &g1,
                                 DataflowGraphView const &g2) {
  DataflowGraph g =
      DataflowGraph::create_copy_of<UnorderedSetDataflowGraph>(g1);
  serial_extend_unsafe(g, g2);
  return g;
}

DataflowGraph parallel_composition(DataflowGraphView const &g1,
                                   DataflowGraphView const &g2) {
  DataflowGraph g =
      DataflowGraph::create_copy_of<UnorderedSetDataflowGraph>(g1);
  parallel_extend_unsafe(g, g2);
  return g;
}

DataflowGraph dataflow_graph_from_sp_decomposition(
    SerialParallelDecomposition const &sp_decomposition) {
  // TODO(@lockshaw): see existing concerns about serial_extend_unsafe

  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
