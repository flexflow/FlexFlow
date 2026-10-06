#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_INSTANCES_UNORDERED_SET_KWARG_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_INSTANCES_UNORDERED_SET_KWARG_DATAFLOW_GRAPH_H

#include "utils/containers/generate_map.h"
#include "utils/containers/set_union.h"
#include "utils/containers/values.h"
#include "utils/graph/biindex.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_all_kwarg_dataflow_edges.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_all_kwarg_dataflow_outputs.h"
#include "utils/graph/kwarg_dataflow_graph/i_kwarg_dataflow_graph.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_edge_query.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_output_query.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/node/node_source.h"

namespace FlexFlow {

template <typename SlotName>
struct UnorderedSetKwargDataflowGraph final
    : public IKwargDataflowGraph<SlotName> {
  UnorderedSetKwargDataflowGraph() = default;

  KwargNodeAddedResult<SlotName>
      add_node(std::map<SlotName, KwargDataflowOutput<SlotName>> const &inputs,
               std::set<SlotName> const &output_slots) override {

    Node new_node = this->node_source.new_node();

    std::map<SlotName, KwargDataflowOutput<SlotName>> outputs = generate_map(
        output_slots,
        [&](SlotName const &output_slot) -> KwargDataflowOutput<SlotName> {
          KwargDataflowOutput<SlotName> output = KwargDataflowOutput<SlotName>{
              /*node=*/new_node,
              /*slot_name=*/output_slot,
          };

          return output;
        });

    this->add_node_unsafe(new_node, inputs, outputs);

    return KwargNodeAddedResult<SlotName>{
        /*node=*/new_node,
        /*outputs=*/outputs,
    };
  }

  void add_node_unsafe(
      Node const &node,
      std::map<SlotName, KwargDataflowOutput<SlotName>> const &inputs,
      std::map<SlotName, KwargDataflowOutput<SlotName>> const &outputs)
      override {
    this->nodes.insert(node);

    for (auto const &[input_slot_name, src] : inputs) {
      KwargDataflowInput<SlotName> dst = KwargDataflowInput<SlotName>{
          node,
          input_slot_name,
      };

      KwargDataflowEdge<SlotName> in_edge = KwargDataflowEdge{
          /*src=*/src,
          /*dst=*/dst,
      };

      this->add_edge(in_edge);
    }

    for (auto const &[_, o] : outputs) {
      this->add_output(o);
    }
  }

  std::set<Node> query_nodes(NodeQuery const &q) const override {
    return apply_query(q.nodes, this->nodes);
  }

  std::set<KwargDataflowEdge<SlotName>>
      query_edges(KwargDataflowEdgeQuery<SlotName> const &q) const override {

    auto is_invariant = [](auto const &qq) -> bool {
      return is_matchall(qq) || is_matchnone(qq);
    };

    bool internal_is_src_slot_invariant = is_invariant(q.src_slots);
    bool internal_is_dst_slot_invariant = is_invariant(q.dst_slots);

    std::set<KwargDataflowEdge<SlotName>> result =
        this->edge_index.query(q.src_nodes, q.dst_nodes);

    if (!internal_is_src_slot_invariant || !internal_is_dst_slot_invariant) {
      result = filter(result, [&](auto const &e) {
        return kwarg_dataflow_edge_query_includes(q, e);
      });
    }

    return result;
  }

  std::set<KwargDataflowOutput<SlotName>> query_outputs(
      KwargDataflowOutputQuery<SlotName> const &q) const override {
    return this->output_index.query(q.nodes, q.output_idxs);
  }

  void clear() {
    this->nodes.clear();
    this->edge_index.clear();
    this->output_index.clear();
  }

  void inplace_materialize_from(
      KwargDataflowGraphView<SlotName> const &v) override {

    this->nodes = get_nodes(v);

    for (KwargDataflowEdge<SlotName> const &e :
         get_all_kwarg_dataflow_edges(v)) {
      this->add_edge(e);
    }

    for (KwargDataflowOutput<SlotName> const &o :
         get_all_kwarg_dataflow_outputs(v)) {
      this->add_output(o);
    }
  };

  UnorderedSetKwargDataflowGraph *clone() const override {
    return new UnorderedSetKwargDataflowGraph<SlotName>{
        this->node_source,
        this->nodes,
        this->edge_index,
        this->output_index,
    };
  }

private:
  void add_edge(KwargDataflowEdge<SlotName> const &e) {
    this->edge_index.add_value(e.src.node, e.dst.node, e);
  }

  void add_output(KwargDataflowOutput<SlotName> const &o) {
    this->output_index.add_value(o.node, o.slot_name, o);
  }

  UnorderedSetKwargDataflowGraph(
      NodeSource const &node_source,
      std::set<Node> const &nodes,
      BiIndex<Node, Node, KwargDataflowEdge<SlotName>> const &edge_index,
      BiIndex<Node, SlotName, KwargDataflowOutput<SlotName>> const
          &output_index)
      : node_source(node_source), nodes(nodes), edge_index(edge_index),
        output_index(output_index) {}

private:
  NodeSource node_source;

  std::set<Node> nodes;

  BiIndex<Node, Node, KwargDataflowEdge<SlotName>> edge_index;
  BiIndex<Node, SlotName, KwargDataflowOutput<SlotName>> output_index;
};

} // namespace FlexFlow

#endif
