#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_VIEW_AS_CLOSED_KWARG_DATAFLOW_GRAPH_BY_MATERIALIZING_INPUTS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_VIEW_AS_CLOSED_KWARG_DATAFLOW_GRAPH_BY_MATERIALIZING_INPUTS_H

#include "utils/bidict/algorithms/right_entries.h"
#include "utils/bidict/generate_bidict.h"
#include "utils/containers/set_union.h"
#include "utils/containers/transform.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_all_kwarg_dataflow_edges.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/view_from_kwarg_dataflow_graph_data.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_edge.dtg.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_graph_view.h"
#include "utils/graph/node/node_source.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_open_kwarg_dataflow_graph_data.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_graph_view.h"
#include "utils/json/optional.h"
#include "utils/overload.h"

namespace FlexFlow {

template <typename GraphInputName, typename SlotName>
std::pair<KwargDataflowGraphView<std::optional<SlotName>>,
          bidict<KwargDataflowGraphInput<GraphInputName>, Node>>
    view_as_closed_kwarg_dataflow_graph_by_materializing_inputs(
        OpenKwargDataflowGraphView<GraphInputName, SlotName> const &g) {
  OpenKwargDataflowGraphData<GraphInputName, SlotName> open_g_data =
      get_open_kwarg_dataflow_graph_data(g);

  NodeSource n;

  bidict<KwargDataflowGraphInput<GraphInputName>, Node> graph_input_nodes =
      generate_bidict(
          open_g_data.inputs,
          [&](KwargDataflowGraphInput<GraphInputName> const &) -> Node {
            return n.new_node();
          });

  auto kwarg_dataflow_output_for_graph_input =
      [&](KwargDataflowGraphInput<GraphInputName> const &i)
      -> KwargDataflowOutput<std::optional<SlotName>> {
    return KwargDataflowOutput<std::optional<SlotName>>{
        /*node=*/graph_input_nodes.at_l(i),
        /*slot_name=*/std::nullopt,
    };
  };

  auto convert_kwarg_dataflow_output =
      [&](KwargDataflowOutput<SlotName> const &o)
      -> KwargDataflowOutput<std::optional<SlotName>> {
    return KwargDataflowOutput<std::optional<SlotName>>{
        /*node=*/o.node,
        /*slot_name=*/o.slot_name,
    };
  };

  auto convert_kwarg_dataflow_input = [&](KwargDataflowInput<SlotName> const &i)
      -> KwargDataflowInput<std::optional<SlotName>> {
    return KwargDataflowInput<std::optional<SlotName>>{
        /*node=*/i.node,
        /*slot_name=*/i.slot_name,
    };
  };

  auto convert_standard_edge = [&](KwargDataflowEdge<SlotName> const &e)
      -> KwargDataflowEdge<std::optional<SlotName>> {
    return KwargDataflowEdge<std::optional<SlotName>>{
        /*src=*/convert_kwarg_dataflow_output(e.src),
        /*dst=*/convert_kwarg_dataflow_input(e.dst),
    };
  };

  auto convert_input_edge =
      [&](KwargDataflowInputEdge<GraphInputName, SlotName> const &e)
      -> KwargDataflowEdge<std::optional<SlotName>> {
    return KwargDataflowEdge<std::optional<SlotName>>{
        /*src=*/kwarg_dataflow_output_for_graph_input(e.src),
        /*dst=*/convert_kwarg_dataflow_input(e.dst),
    };
  };

  auto convert_edge =
      [&](OpenKwargDataflowEdge<GraphInputName, SlotName> const &open_edge)
      -> KwargDataflowEdge<std::optional<SlotName>> {
    return open_edge.template visit<KwargDataflowEdge<std::optional<SlotName>>>(
        overload{
            convert_standard_edge,
            convert_input_edge,
        });
  };

  KwargDataflowGraphData<std::optional<SlotName>> closed_g_data =
      KwargDataflowGraphData<std::optional<SlotName>>{
          /*nodes=*/
          set_union(open_g_data.nodes, right_entries(graph_input_nodes)),
          /*edges=*/transform(open_g_data.edges, convert_edge),
          /*outputs=*/
          set_union(
              transform(open_g_data.outputs, convert_kwarg_dataflow_output),
              transform(open_g_data.inputs,
                        kwarg_dataflow_output_for_graph_input)),
      };

  ASSERT(closed_g_data.edges.size() == open_g_data.edges.size());

  KwargDataflowGraphView<std::optional<SlotName>> closed_g =
      view_from_kwarg_dataflow_graph_data(closed_g_data);

  ASSERT(closed_g_data.edges == get_all_kwarg_dataflow_edges(closed_g));

  return std::pair{
      closed_g,
      graph_input_nodes,
  };
}

} // namespace FlexFlow

#endif
