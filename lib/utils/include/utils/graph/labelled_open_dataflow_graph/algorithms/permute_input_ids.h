#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_ALGORITHMS_PERMUTE_INPUT_IDS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_ALGORITHMS_PERMUTE_INPUT_IDS_H

#include "utils/graph/labelled_open_dataflow_graph/labelled_open_dataflow_graph_view.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_open_dataflow_values.h"
#include "utils/graph/open_dataflow_graph/algorithms/new_dataflow_graph_input.dtg.h"
#include "utils/graph/open_dataflow_graph/algorithms/permute_input_ids.h"
#include "utils/overload.h"

namespace FlexFlow {

template <typename NodeLabel, typename ValueLabel>
LabelledOpenDataflowGraphView<NodeLabel, ValueLabel> permute_input_ids(
  LabelledOpenDataflowGraphView<NodeLabel, ValueLabel> const &g,
  bidict<NewDataflowGraphInput, DataflowGraphInput> const &input_mapping) {
  
  OpenDataflowGraphView permuted = permute_input_ids(
    static_cast<OpenDataflowGraphView>(g),
    input_mapping
  );

  auto old_value_from_new = [&](OpenDataflowValue const &new_value) {
    return new_value.visit<OpenDataflowValue>(overload {
      [](DataflowOutput const &o) {
        return OpenDataflowValue{o};
      },
      [&](DataflowGraphInput const &new_i) {
        return OpenDataflowValue{
          input_mapping.at_l(NewDataflowGraphInput{new_i}),
        };
      },
    });
  };

  std::unordered_map<Node, NodeLabel> node_labels = generate_map(get_nodes(permuted),
                                                                 [&](Node const &n) { 
                                                                   return g.at(n);
                                                                  });

  std::unordered_map<OpenDataflowValue, ValueLabel> value_labels = generate_map(get_open_dataflow_values(permuted),
                                                                                [&](OpenDataflowValue const &new_value) {
                                                                                  return g.at(old_value_from_new(new_value));
                                                                                });

  return with_labelling(permuted, node_labels, value_labels);
}

} // namespace FlexFlow

#endif
