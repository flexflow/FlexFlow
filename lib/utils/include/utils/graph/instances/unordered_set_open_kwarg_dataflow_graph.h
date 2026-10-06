#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_INSTANCES_UNORDERED_SET_OPEN_KWARG_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_INSTANCES_UNORDERED_SET_OPEN_KWARG_DATAFLOW_GRAPH_H

#include "utils/containers/generate_map.h"
#include "utils/graph/biindex.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_output_query.h"
#include "utils/graph/node/node_source.h"
#include "utils/graph/open_kwarg_dataflow_graph/i_open_kwarg_dataflow_graph.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_edge.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_edge_query.h"
#include "utils/graph/query_set.h"

namespace FlexFlow {

template <typename GraphInputName, typename SlotName>
struct UnorderedSetOpenKwargDataflowGraph final
    : public IOpenKwargDataflowGraph<GraphInputName, SlotName> {
  UnorderedSetOpenKwargDataflowGraph() = default;

  KwargNodeAddedResult<SlotName> add_node(
      std::map<SlotName, OpenKwargDataflowValue<GraphInputName, SlotName>> const
          &inputs,
      std::set<SlotName> const &output_slots) override {
    Node new_node = this->node_source.new_node();
    this->nodes.insert(new_node);

    for (auto const &[input_slot_name, input_val] : inputs) {
      KwargDataflowInput<SlotName> dst = KwargDataflowInput<SlotName>{
          new_node,
          input_slot_name,
      };

      OpenKwargDataflowEdge<GraphInputName, SlotName> in_edge =
          mk_open_kwarg_dataflow_edge_from_src_val_and_dst(input_val, dst);

      this->add_edge(in_edge);
    }

    std::map<SlotName, KwargDataflowOutput<SlotName>> outputs = generate_map(
        output_slots,
        [&](SlotName const &output_slot) -> KwargDataflowOutput<SlotName> {
          KwargDataflowOutput<SlotName> output = KwargDataflowOutput<SlotName>{
              /*node=*/new_node,
              /*slot_name=*/output_slot,
          };

          this->add_output(output);

          return output;
        });

    return KwargNodeAddedResult<SlotName>{
        /*node=*/new_node,
        /*outputs=*/outputs,
    };
  }

  KwargDataflowGraphInput<GraphInputName>
      add_input(GraphInputName const &name) override {
    KwargDataflowGraphInput<GraphInputName> input =
        KwargDataflowGraphInput{name};

    this->graph_inputs.insert(input);

    return input;
  }

  std::set<Node> query_nodes(NodeQuery const &q) const override {
    return apply_query(q.nodes, this->nodes);
  }

  std::set<OpenKwargDataflowEdge<GraphInputName, SlotName>>
      query_edges(OpenKwargDataflowEdgeQuery<GraphInputName, SlotName> const &q)
          const override {

    std::set<OpenKwargDataflowEdge<GraphInputName, SlotName>> result;

    auto is_invariant = [](auto const &qq) -> bool {
      return is_matchall(qq) || is_matchnone(qq);
    };

    bool internal_is_src_slot_invariant =
        is_invariant(q.standard_edge_query.src_slots);
    bool internal_is_dst_slot_invariant =
        is_invariant(q.standard_edge_query.dst_slots);

    std::set<OpenKwargDataflowEdge<GraphInputName, SlotName>> standard_edges =
        this->standard_edges_index.query(q.standard_edge_query.src_nodes,
                                         q.standard_edge_query.dst_nodes);

    if (!internal_is_src_slot_invariant || !internal_is_dst_slot_invariant) {
      standard_edges = filter(standard_edges, [&](auto const &e) {
        return open_kwarg_dataflow_edge_query_includes(q, e);
      });
    }

    extend(result, standard_edges);

    std::set<OpenKwargDataflowEdge<GraphInputName, SlotName>> external_edges =
        this->external_edges_index.query(q.input_edge_query.srcs,
                                         q.input_edge_query.dst_nodes);

    bool external_is_dst_slot_invariant =
        is_invariant(q.input_edge_query.dst_slots);
    if (!external_is_dst_slot_invariant) {
      external_edges = filter(external_edges, [&](auto const &e) {
        return open_kwarg_dataflow_edge_query_includes(q, e);
      });
    }

    extend(result, external_edges);

    return result;
  }

  std::set<KwargDataflowOutput<SlotName>> query_outputs(
      KwargDataflowOutputQuery<SlotName> const &q) const override {
    return this->output_index.query(q.nodes, q.output_idxs);
  }

  std::set<KwargDataflowGraphInput<GraphInputName>>
      get_inputs() const override {
    return this->graph_inputs;
  }

  UnorderedSetOpenKwargDataflowGraph *clone() const override {
    return new UnorderedSetOpenKwargDataflowGraph<GraphInputName, SlotName>{
        this->node_source,
        this->graph_inputs,
        this->nodes,
        this->standard_edges_index,
        this->external_edges_index,
        this->output_index,
    };
  }

private:
  UnorderedSetOpenKwargDataflowGraph(
      NodeSource const &node_source,
      std::set<KwargDataflowGraphInput<GraphInputName>> const &graph_inputs,
      std::set<Node> const &nodes,
      BiIndex<Node, Node, OpenKwargDataflowEdge<GraphInputName, SlotName>> const
          &standard_edges_index,
      BiIndex<GraphInputName,
              Node,
              OpenKwargDataflowEdge<GraphInputName, SlotName>> const
          &external_edges_index,
      BiIndex<Node, SlotName, KwargDataflowOutput<SlotName>> const
          &output_index)
      : node_source(node_source), graph_inputs(graph_inputs), nodes(nodes),
        standard_edges_index(standard_edges_index),
        external_edges_index(external_edges_index), output_index(output_index) {
  }

  void
      add_edge(OpenKwargDataflowEdge<GraphInputName, SlotName> const &in_edge) {
    if (in_edge.is_internal_edge()) {
      KwargDataflowEdge<SlotName> const &e = in_edge.require_internal_edge();

      this->standard_edges_index.add_value(e.src.node, e.dst.node, in_edge);
    } else {
      KwargDataflowInputEdge<GraphInputName, SlotName> const &e =
          in_edge.require_input_edge();

      this->external_edges_index.add_value(e.src.name, e.dst.node, in_edge);
    }
  }

  void add_output(KwargDataflowOutput<SlotName> const &o) {
    this->output_index.add_value(o.node, o.slot_name, o);
  }

private:
  NodeSource node_source;

  std::set<Node> nodes;

  std::set<KwargDataflowGraphInput<GraphInputName>> graph_inputs;

  BiIndex<Node, Node, OpenKwargDataflowEdge<GraphInputName, SlotName>>
      standard_edges_index;
  BiIndex<GraphInputName, Node, OpenKwargDataflowEdge<GraphInputName, SlotName>>
      external_edges_index;
  BiIndex<Node, SlotName, KwargDataflowOutput<SlotName>> output_index;
};

} // namespace FlexFlow

#endif
