#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_OPEN_KWARG_DATAFLOW_GRAPH_SUBGRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_OPEN_KWARG_DATAFLOW_GRAPH_SUBGRAPH_H

#include "utils/bidict/generate_bidict.h"
#include "utils/containers/set_of.h"
#include "utils/containers/set_union.h"
#include "utils/containers/values.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_output_query.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_open_kwarg_dataflow_subgraph_incoming_edges.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_open_kwarg_dataflow_subgraph_inputs.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/open_kwarg_dataflow_graph_data.dtg.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/open_kwarg_dataflow_subgraph_result.dtg.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/view_from_open_kwarg_dataflow_graph_data.h"
#include "utils/overload.h"

namespace FlexFlow {

template <typename GraphInputName, typename SlotName>
OpenKwargDataflowSubgraphResult<GraphInputName, SlotName>
    get_open_kwarg_dataflow_graph_subgraph(
        OpenKwargDataflowGraphView<GraphInputName, SlotName> const &g,
        std::set<Node> const &subgraph_nodes,
        std::function<GraphInputName()> const &input_source) {

  std::set<OpenKwargDataflowEdge<GraphInputName, SlotName>>
      subgraph_incoming_edges =
          get_open_kwarg_dataflow_subgraph_incoming_edges(g, subgraph_nodes);

  bidict<OpenKwargDataflowValue<GraphInputName, SlotName>,
         KwargDataflowGraphInput<GraphInputName>>
      full_graph_values_to_subgraph_inputs =
          get_full_kwarg_dataflow_graph_values_to_subgraph_inputs(
              /*g=*/g,
              /*subgraph_nodes=*/subgraph_nodes,
              /*incoming_edges=*/subgraph_incoming_edges,
              /*input_source,=*/input_source);

  OpenKwargDataflowGraphData<GraphInputName, SlotName> subgraph_data =
      get_open_kwarg_dataflow_subgraph_data(
          /*g=*/g,
          /*subgraph_nodes=*/subgraph_nodes,
          /*full_graph_values_to_subgraph_inputs=*/
          full_graph_values_to_subgraph_inputs,
          /*incoming_edges=*/subgraph_incoming_edges);

  return OpenKwargDataflowSubgraphResult{
      view_from_open_kwarg_dataflow_graph_data(subgraph_data),
      full_graph_values_to_subgraph_inputs,
  };
}

template <typename GraphInputName, typename SlotName>
bidict<OpenKwargDataflowValue<GraphInputName, SlotName>,
       KwargDataflowGraphInput<GraphInputName>>
    get_full_kwarg_dataflow_graph_values_to_subgraph_inputs(
        OpenKwargDataflowGraphView<GraphInputName, SlotName> const &g,
        std::set<Node> const &subgraph_nodes,
        std::set<OpenKwargDataflowEdge<GraphInputName, SlotName>> const
            &incoming_edges,
        std::function<GraphInputName()> const &input_source) {

  std::set<OpenKwargDataflowValue<GraphInputName, SlotName>> inputs =
      transform(incoming_edges,
                [](OpenKwargDataflowEdge<GraphInputName, SlotName> const &e) {
                  return get_src_of_open_kwarg_dataflow_edge(e);
                });

  return generate_bidict(
      inputs,
      [&](OpenKwargDataflowValue<GraphInputName, SlotName> const &v)
          -> KwargDataflowGraphInput<GraphInputName> {
        return v.template visit<KwargDataflowGraphInput<GraphInputName>>(
            overload{
                [](KwargDataflowGraphInput<GraphInputName> const &i) {
                  return i;
                },
                [&](KwargDataflowOutput<SlotName> const &) {
                  return KwargDataflowGraphInput<GraphInputName>{
                      input_source(),
                  };
                },
            });
      });
}

template <typename GraphInputName, typename SlotName>
OpenKwargDataflowGraphData<GraphInputName, SlotName>
    get_open_kwarg_dataflow_subgraph_data(
        OpenKwargDataflowGraphView<GraphInputName, SlotName> const &g,
        std::set<Node> const &subgraph_nodes,
        bidict<OpenKwargDataflowValue<GraphInputName, SlotName>,
               KwargDataflowGraphInput<GraphInputName>> const
            &full_graph_values_to_subgraph_inputs,
        std::set<OpenKwargDataflowEdge<GraphInputName, SlotName>> const
            &incoming_edges) {

  std::set<OpenKwargDataflowEdge<GraphInputName, SlotName>>
      subgraph_input_edges = transform(
          incoming_edges,
          [&](OpenKwargDataflowEdge<GraphInputName, SlotName> const &edge) {
            return edge.template visit<
                OpenKwargDataflowEdge<GraphInputName, SlotName>>(overload{
                [&](KwargDataflowInputEdge<GraphInputName, SlotName> const &e)
                    -> OpenKwargDataflowEdge<GraphInputName, SlotName> {
                  return OpenKwargDataflowEdge<GraphInputName, SlotName>{
                      KwargDataflowInputEdge<GraphInputName, SlotName>{
                          full_graph_values_to_subgraph_inputs.at_l(
                              OpenKwargDataflowValue<GraphInputName, SlotName>{
                                  e.src}),
                          e.dst},
                  };
                },
                [&](KwargDataflowEdge<SlotName> const &e) {
                  return OpenKwargDataflowEdge<GraphInputName, SlotName>{
                      KwargDataflowInputEdge<GraphInputName, SlotName>{
                          full_graph_values_to_subgraph_inputs.at_l(
                              OpenKwargDataflowValue<GraphInputName, SlotName>{
                                  e.src}),
                          e.dst,
                      },
                  };
                },
            });
          });

  OpenKwargDataflowEdgeQuery<GraphInputName, SlotName>
      subgraph_interior_edges_query =
          OpenKwargDataflowEdgeQuery<GraphInputName, SlotName>{
              KwargDataflowInputEdgeQuery<GraphInputName, SlotName>{
                  /*srcs=*/query_set<GraphInputName>::match_none(),
                  /*dst_nodes=*/query_set<Node>::match_none(),
                  /*dst_slots=*/query_set<SlotName>::match_none(),
              },
              KwargDataflowEdgeQuery<SlotName>{
                  /*srcs=*/query_set<Node>::match_values_in(subgraph_nodes),
                  /*src_slots=*/query_set<SlotName>::matchall(),
                  /*dsts=*/query_set<Node>::match_values_in(subgraph_nodes),
                  /*dst_slots=*/query_set<SlotName>::matchall(),
              },
          };

  std::set<OpenKwargDataflowEdge<GraphInputName, SlotName>>
      subgraph_interior_edges = g.query_edges(subgraph_interior_edges_query);

  std::set<KwargDataflowGraphInput<GraphInputName>> subgraph_inputs =
      set_of(values(full_graph_values_to_subgraph_inputs));

  std::set<KwargDataflowOutput<SlotName>> subgraph_outputs =
      filter(g.query_outputs(kwarg_dataflow_output_query_all<SlotName>()),
             [&](KwargDataflowOutput<SlotName> const &o) {
               return contains(subgraph_nodes, o.node);
             });

  return OpenKwargDataflowGraphData<GraphInputName, SlotName>{
      /*nodes=*/subgraph_nodes,
      /*edges=*/set_union(subgraph_input_edges, subgraph_interior_edges),
      /*inputs=*/subgraph_inputs,
      /*outputs=*/subgraph_outputs,
  };
}

} // namespace FlexFlow

#endif
