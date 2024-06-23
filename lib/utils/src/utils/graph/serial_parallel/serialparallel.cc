#include "utils/graph/serial_parallel/serialparallel.h"
#include "./serialparallel_internal.h"
#include "./sink_settings.dtg.h" 
#include "./source_settings.dtg.h"
#include "utils/containers.h"
#include "utils/graph/algorithms.h"

namespace FlexFlow {

Node find_source_node(DiGraphView const &g) {
  std::unordered_set<Node> srcs = get_sources(g);
  return get_only(srcs);
}

Node find_sink_node(DiGraphView const &g) {
  std::unordered_set<Node> sinks = get_sinks(g);
  return get_only(sinks);
}

std::optional<Node> find_bottleneck_node(DiGraphView const &g) {
  std::unordered_set<Node> sources = get_sources(g);
  std::unordered_set<Node> sinks = get_sinks(g);

  std::optional<Node> maybe_bottleneck = get_imm_post_dominator(g, sources);
  if (maybe_bottleneck.has_value()) {
    assert(contains(get_dominators(g, sinks), maybe_bottleneck.value()));
  }
  return maybe_bottleneck;
}

std::unordered_set<Node> from_source_to_sink(DiGraphView const &g,
                                             Node const &src,
                                             Node const &sink) {
  assert(contains(get_dominators(g, sink), src));

  std::vector<Node> bfs = get_bfs_ordering(g, {src});
  auto end = find(bfs, sink);
  assert(end != bfs.end());

  std::unordered_set<Node> result(bfs.cbegin(), ++end);
  return result;
}

SerialParallelDecomposition
    get_serial_parallel_decomposition(DiGraphView const &g) {
  std::variant<IntermediateSpDecompositionTree, Node> ast = sp_decomposition(g);
  return to_final_ast(ast);
}

std::unordered_set<Node> get_nodes(SerialParallelDecomposition const &sp) {
  return sp.visit<std::unordered_set<Node>>([](auto &&t) { return get_nodes(t); });
}

std::unordered_set<Node> get_nodes(Serial const &serial) {
  return set_union(transform(
      serial.children,
      [](std::variant<Parallel, Node> const &child) -> std::unordered_set<Node> {
        return std::visit([](auto &&t) { return get_nodes(t); }, child);
      }));
}

std::unordered_set<Node> get_nodes(Parallel const &parallel) {
  return set_union(
      transform(parallel.children, [](std::variant<Serial, Node> const &child) {
        return std::visit([](auto &&t) { return get_nodes(t); }, child);
      }));
}

std::unordered_set<Node> get_nodes(Node const &node) {
  return {node};
}

// std::unordered_map<Node, Node> parallel_extend(MultiDiGraph &g,
//                                                MultiDiGraph const &ext) {
//   std::unordered_map<Node, Node> node_map;
//   std::unordered_map<NodePort, NodePort> node_port_map;
//   for (Node const &node : get_nodes(MultiDiGraphView(ext))) {
//     node_map.emplace(node, g.add_node());
//   }
//   for (NodePort const &node_port : get_present_node_ports(ext)) {
//     node_port_map.emplace(node_port, g.add_node_port());
//   }
//   for (MultiDiEdge const &edge : get_edges(ext)) {
//     g.add_edge(MultiDiEdge{node_map.at(edge.dst),
//                            node_port_map.at(edge.dst_idx),
//                            node_map.at(edge.src),
//                            node_port_map.at(edge.src_idx)});
//   }
//   return node_map;
// }

// std::unordered_map<Node, Node> serial_extend(MultiDiGraph &g,
//                                              MultiDiGraph const &ext) {
//   std::unordered_set<Node> original_sinks = get_sinks(g);
//   std::unordered_map<Node, Node> node_map = parallel_extend(g, ext);
//   for (Node const &node1 : original_sinks) {
//     for (Node const &node2 : get_sources(ext)) {
//       g.add_edge(MultiDiEdge{
//           node_map.at(node2), g.add_node_port(), node1, g.add_node_port()});
//     }
//   }
//   return node_map;
// }

// MultiDiGraph serial_composition(MultiDiGraph const &g1,
//                                 MultiDiGraph const &g2) {
//   MultiDiGraph g = g1;
//   serial_extend(g, g2);
//   return g;
// }

// MultiDiGraph parallel_composition(MultiDiGraph const &g1,
//                                   MultiDiGraph const &g2) {
//   MultiDiGraph g = g1;
//   parallel_extend(g, g2);
//   return g;
// }

// struct MultiDiGraphFromSPDecompositionFunctor {
//   template <typename T>
//   MultiDiGraph operator()(T const &t) {
//     return multidigraph_from_sp_decomposition(t);
//   }
// };

// MultiDiGraph multidigraph_from_sp_decomposition(
//     SerialParallelDecomposition const &sp_decomposition) {
//   return visit(MultiDiGraphFromSPDecompositionFunctor{}, sp_decomposition);
// }

// MultiDiGraph multidigraph_from_sp_decomposition(
//     std::variant<Parallel, Node> const &sp_decomposition) {
//   return visit(MultiDiGraphFromSPDecompositionFunctor{}, sp_decomposition);
// }

// MultiDiGraph multidigraph_from_sp_decomposition(
//     std::variant<Serial, Node> const &sp_decomposition) {
//   return visit(MultiDiGraphFromSPDecompositionFunctor{}, sp_decomposition);
// }

// MultiDiGraph multidigraph_from_sp_decomposition(Serial const &serial) {
//   MultiDiGraph g = MultiDiGraph::create<AdjacencyMultiDiGraph>();
//   for (std::variant<Parallel, Node> const &child : serial.children) {
//     serial_extend(g, multidigraph_from_sp_decomposition(child));
//   }
//   return g;
// }

// MultiDiGraph multidigraph_from_sp_decomposition(Parallel const &parallel) {
//   MultiDiGraph g = MultiDiGraph::create<AdjacencyMultiDiGraph>();
//   for (std::variant<Serial, Node> const &child : parallel.children) {
//     parallel_extend(g, multidigraph_from_sp_decomposition(child));
//   }
//   return g;
// }

// MultiDiGraph multidigraph_from_sp_decomposition(Node const &Node) {
//   MultiDiGraph g = MultiDiGraph::create<AdjacencyMultiDiGraph>();
//   g.add_node();
//   return g;
// }

} // namespace FlexFlow
