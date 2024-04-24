#include "pcg/file_format/v1/graphs.h"
#include "utils/graph/algorithms.h"

namespace FlexFlow {

static V1MultiDiGraph to_v1(MultiDiGraphView const &g,
                     bidict<Node, size_t> const &nodes,
                     bidict<NodePort, size_t> const &node_ports) {
  std::unordered_set<V1GraphEdge> edges;
  for (MultiDiEdge const &e : get_edges(g)) {
    edges.insert({nodes.at_l(e.src),
                  node_ports.at_l(e.src_idx),
                  nodes.at_l(e.dst),
                  node_ports.at_l(e.dst_idx)});
  }

  return V1MultiDiGraph{
      count(nodes.size()),
      count(node_ports.size()),
      edges,
  };
}

static V1MultiDiGraph to_v1(MultiDiGraphView const &g) {
  return to_v1(g,
               enumerate(get_nodes(g)).reversed(),
               enumerate(get_present_node_ports(g)).reversed());
}

template <typename NodeLabel, typename OutputLabel>
static V1JsonableGraph<NodeLabel, OutputLabel>
    to_v1(OutputLabelledMultiDiGraph<NodeLabel, OutputLabel> const &g) {
  bidict<size_t, Node> nodes = enumerate(get_nodes(g));
  bidict<size_t, NodePort> node_ports = enumerate(get_present_node_ports(g));

  V1MultiDiGraph unlabelled = to_v1(g, nodes.reversed(), node_ports.reversed());
  std::unordered_map<size_t, NodeLabel> node_labels =
      map_values(nodes, [&](Node const &n) { return g.at(n); });

  bidict<size_t, MultiDiOutput> outputs_bidict = enumerate(get_outputs(g));
  std::unordered_map<size_t, V1GraphOutput> outputs =
      map_values(outputs_bidict, [&](MultiDiOutput const &o) {
        return V1GraphOutput{nodes.at_r(o.src), node_ports.at_r(o.src_idx)};
      });

  std::unordered_map<size_t, OutputLabel> output_labels = map_values(
      outputs_bidict, [&](MultiDiOutput const &o) { return g.at(o); });

  return {node_labels, outputs, output_labels, unlabelled};
}

V1ComputationGraph to_v1(ComputationGraph const &g) {
  return to_v1<LayerAttrs, TensorAttrs>(g.raw_graph);
}

V1ParallelComputationGraph to_v1(ParallelComputationGraph const &g) {
  return to_v1<ParallelLayerAttrs, ParallelTensorAttrs>(g.raw_graph);
}

} // namespace FlexFlow
