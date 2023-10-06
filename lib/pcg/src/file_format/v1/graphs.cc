#include "pcg/file_format/v1/graphs.h"
#include "utils/graph/algorithms.h"

namespace FlexFlow {

V1MultiDiGraph to_v1(MultiDiGraphView const &g) {
  return to_v1(g,
               enumerate(get_nodes(g)).reversed(),
               enumerate(get_present_node_ports(g)).reversed());
}

V1MultiDiGraph to_v1(MultiDiGraphView const &g,
                     std::unordered_map<Node, size_t> const &nodes,
                     std::unordered_map<NodePort, size_t> const &node_ports) {
  std::unordered_set<V1GraphEdge> edges;
  for (MultiDiEdge const &e : get_edges(g)) {
    edges.insert({nodes.at(e.src),
                  node_ports.at(e.srcIdx),
                  nodes.at(e.dst),
                  node_ports.at(e.dstIdx)});
  }

  return V1MultiDiGraph{
      count(nodes.size()),
      count(node_ports.size()),
      edges,
  };
}

template <typename NodeLabel, typename OutputLabel>
V1JsonableGraph<decltype(to_v1(std::declval<NodeLabel>())),
                decltype(to_v1(std::declval<OutputLabel>()))>
    to_v1(OutputLabelledMultiDiGraph<NodeLabel, OutputLabel> const &g) {
  using V1NodeLabel = decltype(to_v1(std::declval<NodeLabel>()));
  using V1OutputLabel = decltype(to_v1(std::declval<OutputLabel>()));

  bidict<size_t, Node> nodes = enumerate(get_nodes(g));
  bidict<size_t, NodePort> node_ports = enumerate(get_present_node_ports(g));

  V1MultiDiGraph unlabelled = to_v1(g, nodes.reversed(), node_ports.reversed());
  std::unordered_map<size_t, V1NodeLabel> node_labels =
      map_values(nodes, [&](Node const &n) { return to_v1(g.at(n)); });
  bidict<size_t, MultiDiOutput> outputs_bidict = enumerate(get_outputs(g));
  std::unordered_map<size_t, V1GraphOutput> outputs =
      map_values(outputs_bidict, [&](MultiDiOutput const &o) {
        return V1GraphOutput{nodes.at_r(o.node), node_ports.at_r(o.idx)};
      });
  std::unordered_map<size_t, V1OutputLabel> output_labels = map_values(
      outputs_bidict, [&](MultiDiOutput const &o) { return to_v1(g.at(o)); });

  return {node_labels, outputs, output_labels, unlabelled};
}

V1ComputationGraph to_v1(ComputationGraph const &g) {
  return to_v1<Layer, Tensor>(g.value());
}

} // namespace FlexFlow
