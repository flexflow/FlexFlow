#include "utils/graph/serial_parallel/graph_generation.h"
#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/digraph/digraph.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/instances/unordered_set_dataflow_graph.h"
#include "utils/graph/node/algorithms.h"

namespace FlexFlow {

void parallel_extend_unsafe(DataflowGraph &g, DataflowGraphView const &ext) {
  for (Node const &node : get_nodes(ext)) {
    g.add_node_unsafe(node, get_inputs(ext, node), get_outputs(ext, node));
  }
}

std::unordered_map<Node, Node> parallel_extend(DiGraph &g,
                                               DiGraphView const &ext) {
  std::unordered_map<Node, Node> node_map;
  for (Node const &node : get_nodes(ext)) {
    node_map.emplace(node, g.add_node());
  }
  for (DirectedEdge const &edge : get_edges(ext)) {
    g.add_edge(DirectedEdge{node_map.at(edge.src), node_map.at(edge.dst)});
  }
  return node_map;
}

std::unordered_map<Node, Node> serial_extend(DiGraph &g,
                                             DiGraphView const &ext) {
  std::unordered_set<Node> original_sinks = get_sinks(g);
  std::unordered_map<Node, Node> node_map = parallel_extend(g, ext);
  for (Node const &node1 : original_sinks) {
    for (Node const &node2 : get_sources(ext)) {
      g.add_edge(DirectedEdge{node1, node_map.at(node2)});
    }
  }
  return node_map;
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
