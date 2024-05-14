#include "pcg/file_format/v1/graphs.h"
#include "utils/graph/algorithms.h"
#include "pcg/file_format/v1/graphs/v1_operator_graph.dtg.h"
#include "utils/integer_conversions.h"
#include "pcg/file_format/v1/graphs/v1_multidigraph.h"
#include "pcg/dataflow_graph.h"

namespace FlexFlow {

/* static V1OperatorGraph to_v1(OperatorGraphView const &g, bidict<Node, size_t> const &nodes) { */
/*   std::unordered_set<V1GraphEdge> edges; */
/*   for (MultiDiEdge const &e : get_edges(g)) { */
/*     size_t src_node = nodes.at_l(get_src_node(e)); */
/*     size_t dst_node = nodes.at_l(get_dst_node(e)); */
/*     size_t src_idx = size_t_from_int(get_src_idx(e)); */
/*     size_t dst_idx = size_t_from_int(get_dst_idx(e)); */
/*     V1GraphEdge v1_e = {src_node, src_idx, dst_node, dst_idx}; */
/*     edges.insert(v1_e); */
/*   } */

/*   return V1OperatorGraph{ */
/*       count(nodes.size()), */
/*       edges, */
/*   }; */
/* } */


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

/* static V1MultiDiGraph to_v1(MultiDiGraphView const &g) { */
/*   return to_v1(g, */
/*                enumerate(get_nodes(g)).reversed(), */
/*                enumerate(get_present_node_ports(g)).reversed()); */
/* } */

/* template <typename NodeLabel, typename OutputLabel> */
/* static V1JsonableGraph<NodeLabel, OutputLabel> */ 
/*     to_v1(LabelledOperatorGraphView<NodeLabel, OutputLabel> const &g) { */

/*   bidict<size_t, Node> nodes = enumerate(get_nodes(g)); */

/*   V1OperatorGraph unlabelled = to_v1(g, nodes.reversed()); */
/*   std::unordered_map<size_t, NodeLabel> node_labels = */
/*       map_values(nodes, [&](Node const &n) { return g.at(n); }); */

/*   bidict<size_t, OperatorGraphOutput> outputs_bidict = enumerate(get_outputs(g)); */
/*   std::unordered_map<size_t, V1GraphOutput> outputs = */
/*       map_values(outputs_bidict, [&](OperatorGraphOutput const &o) { */
/*         return V1GraphOutput{nodes.at_r(get_node(o)), size_t_from_int(get_idx(o))}; */
/*       }); */

/*   std::unordered_map<size_t, OutputLabel> output_labels = map_values( */
/*       outputs_bidict, [&](OperatorGraphOutput const &o) { return g.at(o); }); */

/*   return {node_labels, outputs, output_labels, unlabelled}; */
/* } */

template <typename NodeLabel, typename OutputLabel>
static bidict<size_t, NodePort> get_ports_by_idx(DataflowGraph<NodeLabel, OutputLabel> const &g) {
  bidict<size_t, NodePort> result;
  for (NodePort const &p : get_present_node_ports(g.get_raw_graph())) {
    size_t idx = size_t_from_int(g.idx_for_port(p));
    result.equate(idx, p);
  }
  return result;
}

template <typename NodeLabel, typename OutputLabel>
static V1JsonableGraph<NodeLabel, OutputLabel>
    to_v1(DataflowGraph<NodeLabel, OutputLabel> const &g) {

  bidict<size_t, Node> nodes = enumerate(get_nodes(g.get_raw_graph()));
  bidict<size_t, NodePort> node_ports = get_ports_by_idx(g);

  V1MultiDiGraph unlabelled = to_v1(g.get_raw_graph(), nodes.reversed(), node_ports.reversed());
  std::unordered_map<size_t, NodeLabel> node_labels =
      map_values(nodes, [&](Node const &n) { return g.at(n); });

  bidict<size_t, MultiDiOutput> outputs_bidict = enumerate(get_outputs(g.get_raw_graph()));
  std::unordered_map<size_t, V1GraphOutput> outputs =
      map_values(outputs_bidict, [&](MultiDiOutput const &o) {
        return V1GraphOutput{nodes.at_r(o.src), node_ports.at_r(o.src_idx)};
      });

  std::unordered_map<size_t, OutputLabel> output_labels = map_values(
      outputs_bidict, [&](MultiDiOutput const &o) { return g.at(o); });

  return {node_labels, outputs, output_labels, unlabelled};
}

template <typename NodeLabel, typename OutputLabel>
static V1JsonableGraph<NodeLabel, OutputLabel>
    to_v1(OutputLabelledMultiDiGraphView<NodeLabel, OutputLabel> const &g) {
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
