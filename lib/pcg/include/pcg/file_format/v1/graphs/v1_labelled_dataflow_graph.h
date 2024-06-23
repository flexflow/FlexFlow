#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_FILE_FORMAT_V1_GRAPHS_V1_LABELLED_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_FILE_FORMAT_V1_GRAPHS_V1_LABELLED_DATAFLOW_GRAPH_H

#include "pcg/file_format/v1/graphs/v1_labelled_dataflow_graph.dtg.h"
#include "utils/graph/labelled_dataflow_graph/labelled_dataflow_graph_view.h"

namespace FlexFlow {

template <typename NodeLabel, typename OutputLabel>
V1LabelledDataflowGraph<NodeLabel, OutputLabel>
    to_v1(LabelledDataflowGraphView<NodeLabel, OutputLabel> const &g) {

  bidict<size_t, Node> nodes = enumerate(get_nodes(g));

  V1DataflowGraph unlabelled = to_v1(g, nodes.reversed());
  std::unordered_map<size_t, NodeLabel> node_labels =
      map_values(nodes, [&](Node const &n) { return g.at(n); });

  std::unordered_map<size_t, V1GraphOutput> outputs =
      map_values(nodes, [&](MultiDiOutput const &o) {
        return V1GraphOutput{nodes.at_r(o.src), node_ports.at_r(o.src_idx)};
      });

  std::unordered_map<size_t, OutputLabel> output_labels = map_values(
      outputs_bidict, [&](MultiDiOutput const &o) { return g.at(o); });

  return V1JsonableGraph<NodeLabel, OutputLabel>{
      node_labels, outputs, output_labels, unlabelled};
}

} // namespace FlexFlow

#endif
