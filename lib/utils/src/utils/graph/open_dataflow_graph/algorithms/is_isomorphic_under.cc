#include "utils/graph/open_dataflow_graph/algorithms/is_isomorphic_under.h"
#include "utils/graph/node/algorithms/new_node.dtg.h"
#include "utils/graph/open_dataflow_graph/algorithms/new_dataflow_graph_input.dtg.h"
#include "utils/graph/open_dataflow_graph/algorithms/permute_input_ids.h"
#include "utils/graph/open_dataflow_graph/algorithms/permute_node_ids.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_graph_data.h"

namespace FlexFlow {

bool is_isomorphic_under(OpenDataflowGraphView const &src,
                         OpenDataflowGraphView const &dst,
                         OpenDataflowGraphIsomorphism const &candidate_isomorphism) {

  bidict<NewNode, Node> node_permutation = map_values(candidate_isomorphism.node_mapping, 
                                                      [](Node const &dst_node) { return NewNode{dst_node}; }
                                                      ).reversed();
  bidict<NewDataflowGraphInput, DataflowGraphInput> input_permutation = map_values(candidate_isomorphism.input_mapping,
                                                                                   [](DataflowGraphInput const &dst_input) { return NewDataflowGraphInput{dst_input}; }
                                                                                  ).reversed();
  return get_graph_data(permute_input_ids(permute_node_ids(src, node_permutation), input_permutation)) == get_graph_data(dst);
}

} // namespace FlexFlow
