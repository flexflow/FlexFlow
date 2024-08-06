#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_ALGORITHMS_FIND_ISOMORPHISM_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_ALGORITHMS_FIND_ISOMORPHISM_H

#include "utils/containers/zip.h"
#include "utils/containers/as_vector.h"
#include "utils/containers/get_all_permutations.h"
#include "utils/graph/labelled_open_dataflow_graph/labelled_open_dataflow_graph_view.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/is_isomorphic_under.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/open_dataflow_graph_isomorphism.dtg.h"

namespace FlexFlow {

template <typename NodeLabel, typename EdgeLabel>
std::optional<OpenDataflowGraphIsomorphism> find_isomorphism(LabelledOpenDataflowGraphView<NodeLabel, EdgeLabel> const &src,
                                                             LabelledOpenDataflowGraphView<NodeLabel, EdgeLabel> const &dst) {
  std::vector<Node> src_nodes = as_vector(get_nodes(src));
  std::vector<DataflowGraphInput> src_inputs = as_vector(get_inputs(src));
  for (std::vector<Node> const &dst_nodes : get_all_permutations(get_nodes(dst))) {
    for (std::vector<DataflowGraphInput> const &dst_inputs : get_all_permutations(get_inputs(dst))) {
      std::vector<std::pair<Node, Node>> zipped_nodes = zip(src_nodes, dst_nodes);
      bidict<Node, Node> node_mapping{zipped_nodes.cbegin(), zipped_nodes.cend()};

      std::vector<std::pair<DataflowGraphInput, DataflowGraphInput>> zipped_inputs = zip(src_inputs, dst_inputs);
      bidict<DataflowGraphInput, DataflowGraphInput> input_mapping{zipped_inputs.cbegin(), zipped_inputs.cend()};

      OpenDataflowGraphIsomorphism candidate_isomorphism = OpenDataflowGraphIsomorphism{
        node_mapping, 
        input_mapping,
      };

      if (is_isomorphic_under(src, dst, candidate_isomorphism)) {
        return candidate_isomorphism;
      }
    }
  }
  
  return std::nullopt;
}

} // namespace FlexFlow

#endif
