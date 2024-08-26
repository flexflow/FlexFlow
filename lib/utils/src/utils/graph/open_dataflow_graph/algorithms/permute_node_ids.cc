#include "utils/graph/open_dataflow_graph/algorithms/permute_node_ids.h"
#include "utils/bidict/algorithms/right_entries.h"
#include "utils/bidict/bidict.h"
#include "utils/bidict/generate_bidict.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/set_minus.h"
#include "utils/containers/transform.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/node/node_query.h"
#include "utils/graph/node/node_source.h"
#include "utils/graph/open_dataflow_graph/algorithms/from_open_dataflow_graph_data.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_graph_data.h"
#include "utils/graph/query_set.h"
#include "utils/overload.h"

namespace FlexFlow {

OpenDataflowGraphView
    permute_node_ids(OpenDataflowGraphView const &g,
                     bidict<NewNode, Node> const &new_node_tofrom_old_node) {
  auto new_node_from_old = [&](Node const &n) -> Node {
    return new_node_tofrom_old_node.at_r(n).raw_node;
  };

  auto new_output_from_old = [&](DataflowOutput const &o) -> DataflowOutput {
    return DataflowOutput{
        new_node_from_old(o.node),
        o.idx,
    };
  };

  auto new_input_from_old = [&](DataflowInput const &i) -> DataflowInput {
    return DataflowInput{
        new_node_from_old(i.node),
        i.idx,
    };
  };

  auto new_edge_from_old = [&](OpenDataflowEdge const &e) {
    return e.visit<OpenDataflowEdge>(overload{
        [&](DataflowInputEdge const &input_e) {
          return OpenDataflowEdge{
              DataflowInputEdge{
                  input_e.src,
                  new_input_from_old(input_e.dst),
              },
          };
        },
        [&](DataflowEdge const &standard_e) {
          return OpenDataflowEdge{
              DataflowEdge{
                  new_output_from_old(standard_e.src),
                  new_input_from_old(standard_e.dst),
              },
          };
        },
    });
  };

  OpenDataflowGraphData old_data = get_graph_data(g);

  OpenDataflowGraphData permuted_data = OpenDataflowGraphData{
      transform(old_data.nodes, new_node_from_old),
      transform(old_data.edges, new_edge_from_old),
      old_data.inputs,
      transform(old_data.outputs, new_output_from_old),
  };

  return from_open_dataflow_graph_data(permuted_data);
}

} // namespace FlexFlow
