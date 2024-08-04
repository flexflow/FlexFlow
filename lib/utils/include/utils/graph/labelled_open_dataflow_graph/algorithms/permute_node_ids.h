#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_ALGORITHMS_PERMUTE_NODE_IDS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_ALGORITHMS_PERMUTE_NODE_IDS_H

#include "utils/graph/labelled_open_dataflow_graph/labelled_open_dataflow_graph_view.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_open_dataflow_values.h"
#include "utils/graph/open_dataflow_graph/algorithms/permute_node_ids.h"
#include "utils/overload.h"

namespace FlexFlow {

template <typename NodeLabel, typename ValueLabel>
LabelledOpenDataflowGraphView<NodeLabel, ValueLabel> permute_node_ids(
  LabelledOpenDataflowGraphView<NodeLabel, ValueLabel> const &g,
  bidict<NewNode, Node> const &new_node_tofrom_old_node) {
  OpenDataflowGraphView permuted = permute_node_ids(
    static_cast<OpenDataflowGraphView>(g),
    new_node_tofrom_old_node
  );

  auto old_node_from_new = [&](Node const &new_node) {
    return new_node_tofrom_old_node.at_l(NewNode{new_node});
  };

  auto old_value_from_new = [&](OpenDataflowValue const &new_value) {
    return new_value.visit<OpenDataflowValue>(overload {
      [&](DataflowOutput const &new_o) {
        return OpenDataflowValue{
          DataflowOutput{
            old_node_from_new(new_o.node),
            new_o.idx,
          },
        };
      },
      [](DataflowGraphInput const &i) {
        return OpenDataflowValue{i}; 
      },
    });
  };

  std::unordered_map<Node, NodeLabel> node_labels = generate_map(get_nodes(permuted),
                                                                 [&](Node const &new_node) {
                                                                   return g.at(old_node_from_new(new_node));
                                                                 });

  std::unordered_map<OpenDataflowValue, ValueLabel> value_labels = generate_map(get_open_dataflow_values(permuted),
                                                                                [&](OpenDataflowValue const &new_value) {
                                                                                  return g.at(old_value_from_new(new_value));
                                                                                });

  return with_labelling(permuted, node_labels, value_labels);
}

} // namespace FlexFlow

#endif
