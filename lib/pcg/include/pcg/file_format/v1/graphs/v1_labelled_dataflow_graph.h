#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_FILE_FORMAT_V1_GRAPHS_V1_LABELLED_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_FILE_FORMAT_V1_GRAPHS_V1_LABELLED_DATAFLOW_GRAPH_H

#include "pcg/file_format/v1/graphs/v1_dataflow_graph.h"
#include "pcg/file_format/v1/graphs/v1_labelled_dataflow_graph.dtg.h"
#include "utils/bidict/algorithms/bidict_from_enumerating.h"
#include "utils/containers/map_values.h"
#include "utils/containers/transform.h"
#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/labelled_dataflow_graph/labelled_dataflow_graph_view.h"
#include "utils/graph/node/algorithms.h"

namespace FlexFlow {

template <typename NodeLabel, typename OutputLabel>
std::pair<
  V1LabelledDataflowGraph<NodeLabel, OutputLabel>,
  bidict<int, Node>
> to_v1_including_node_numbering(LabelledDataflowGraphView<NodeLabel, OutputLabel> const &g) {

  bidict<int, Node> nodes = bidict_from_enumerating(get_nodes(g));

  V1DataflowGraph unlabelled = to_v1(g, nodes.reversed());

  std::unordered_map<int, NodeLabel> node_labels = map_values(
      nodes.as_unordered_map(), [&](Node const &n) { return g.at(n); });

  std::unordered_map<int, std::vector<OutputLabel>> output_labels =
      map_values(nodes.as_unordered_map(), [&](Node const &n) {
        return transform(get_outputs(g, n),
                         [&](DataflowOutput const &o) { return g.at(o); });
      });

  return {
    V1LabelledDataflowGraph<NodeLabel, OutputLabel>{
      node_labels, output_labels, unlabelled},
   nodes,
  };
}

template <typename NodeLabel, typename OutputLabel>
V1LabelledDataflowGraph<NodeLabel, OutputLabel>
    to_v1(LabelledDataflowGraphView<NodeLabel, OutputLabel> const &g) {
  return to_v1_including_node_numbering(g).first;
}

} // namespace FlexFlow

#endif
