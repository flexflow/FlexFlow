#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_INSTANCES_TASK_SET_OPEN_KWARG_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_INSTANCES_TASK_SET_OPEN_KWARG_DATAFLOW_GRAPH_H

#include "utils/containers/contains_key.h"
#include "utils/containers/enumerate.h"
#include "utils/containers/extend.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/keys.h"
#include "utils/containers/map_values.h"
#include "utils/graph/biindex.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_all_kwarg_dataflow_edges.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_all_kwarg_dataflow_outputs.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_node_added_result.dtg.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/i_labelled_open_kwarg_dataflow_graph.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/i_labelled_open_kwarg_dataflow_graph_view.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/node/node_query.h"
#include "utils/graph/node/node_source.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_all_kwarg_dataflow_graph_inputs.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_all_open_kwarg_dataflow_edges.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_edge.h"
#include "utils/overload.h"

namespace FlexFlow {

template <typename NodeLabel,
          typename ValueLabel,
          typename GraphInputName,
          typename SlotName>
struct UnorderedSetLabelledOpenKwargDataflowGraph final
    : public ILabelledOpenKwargDataflowGraph<NodeLabel,
                                             ValueLabel,
                                             GraphInputName,
                                             SlotName>,
      public ILabelledKwargDataflowGraph<NodeLabel, ValueLabel, SlotName> {
public:
  UnorderedSetLabelledOpenKwargDataflowGraph() = default;

  KwargNodeAddedResult<SlotName>
      add_node(NodeLabel const &node_label,
               std::map<SlotName, KwargDataflowOutput<SlotName>> const &inputs,
               std::map<SlotName, ValueLabel> const &output_labels) override {
    return this->add_node(
        node_label,
        map_values(inputs,
                   [](KwargDataflowOutput<SlotName> const &o) {
                     return OpenKwargDataflowValue<GraphInputName, SlotName>{o};
                   }),
        output_labels);
  };

  KwargNodeAddedResult<SlotName> add_node(
      NodeLabel const &node_label,
      std::map<SlotName, OpenKwargDataflowValue<GraphInputName, SlotName>> const
          &inputs,
      std::map<SlotName, ValueLabel> const &output_labels) override {

    Node new_node = this->node_source.new_node();
    this->nodes.insert(new_node);
    this->node_labels.insert({new_node, node_label});

    for (auto const &[input_slot_name, input_val] : inputs) {
      KwargDataflowInput<SlotName> dst = KwargDataflowInput<SlotName>{
          new_node,
          input_slot_name,
      };

      OpenKwargDataflowEdge<GraphInputName, SlotName> in_edge =
          mk_open_kwarg_dataflow_edge_from_src_val_and_dst(input_val, dst);

      this->add_edge(in_edge);
    }

    std::map<SlotName, KwargDataflowOutput<SlotName>> result_outputs;

    for (auto const &[slot_name, label] : output_labels) {
      KwargDataflowOutput<SlotName> o = KwargDataflowOutput<SlotName>{
          /*node=*/new_node,
          /*slot_name=*/slot_name,
      };

      this->add_output(o, label);

      result_outputs.insert({slot_name, o});
    }

    return KwargNodeAddedResult<SlotName>{
        /*node=*/new_node,
        /*outputs=*/result_outputs,
    };
  }

  KwargDataflowGraphInput<GraphInputName>
      add_input(GraphInputName const &name,
                ValueLabel const &value_label) override {
    KwargDataflowGraphInput<GraphInputName> input =
        KwargDataflowGraphInput{name};

    ASSERT(!contains_key(this->graph_input_labels, input));
    this->graph_input_labels.insert({input, value_label});

    return input;
  }

  std::set<Node> query_nodes(NodeQuery const &q) const override {
    return apply_node_query(q, this->nodes);
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

    std::set<OpenKwargDataflowEdge<GraphInputName, SlotName>> internal_edges =
        this->internal_edges_index.query(q.standard_edge_query.src_nodes,
                                         q.standard_edge_query.dst_nodes);

    if (!internal_is_src_slot_invariant || !internal_is_dst_slot_invariant) {
      internal_edges = filter(internal_edges, [&](auto const &e) {
        return open_kwarg_dataflow_edge_query_includes(q, e);
      });
    }

    extend(result, internal_edges);

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
    return keys(this->graph_input_labels);
  }

  NodeLabel at(Node const &n) const override {
    return this->node_labels.at(n);
  }

  ValueLabel at(OpenKwargDataflowValue<GraphInputName, SlotName> const &v)
      const override {
    return v.template visit<ValueLabel>(overload{
        [&](KwargDataflowOutput<SlotName> const &o) -> ValueLabel {
          return this->output_labels.at(o);
        },
        [&](KwargDataflowGraphInput<GraphInputName> const &gi) -> ValueLabel {
          return this->graph_input_labels.at(gi);
        }});
  }

  void clear() {
    this->nodes.clear();
    this->internal_edges_index.clear();
    this->external_edges_index.clear();
    this->output_index.clear();
    this->graph_input_labels.clear();
    this->node_labels.clear();
    this->output_labels.clear();
  }

  void inplace_materialize_from(
      LabelledKwargDataflowGraphView<NodeLabel, ValueLabel, SlotName> const
          &view) override {

    std::set<Node> view_nodes = get_nodes(view);
    std::set<KwargDataflowEdge<SlotName>> view_edges =
        get_all_kwarg_dataflow_edges(view);
    std::set<KwargDataflowOutput<SlotName>> view_outputs =
        get_all_kwarg_dataflow_outputs(view);

    this->clear();

    this->nodes = view_nodes;
    this->node_labels =
        generate_map(view_nodes, [&](Node const &n) { return view.at(n); });

    for (KwargDataflowEdge<SlotName> const &e : view_edges) {
      this->add_edge(OpenKwargDataflowEdge<GraphInputName, SlotName>{e});
    }

    for (KwargDataflowOutput<SlotName> const &o : view_outputs) {
      this->add_output(o, view.at(o));
    }
  }

  void inplace_materialize_from(
      LabelledOpenKwargDataflowGraphView<NodeLabel,
                                         ValueLabel,
                                         GraphInputName,
                                         SlotName> const &view) override {
    std::set<KwargDataflowGraphInput<GraphInputName>> view_inputs =
        get_all_kwarg_dataflow_graph_inputs(view);
    std::set<Node> view_nodes = get_nodes(view);
    std::set<OpenKwargDataflowEdge<GraphInputName, SlotName>> view_edges =
        get_all_open_kwarg_dataflow_edges(view);
    std::set<KwargDataflowOutput<SlotName>> view_outputs =
        get_all_kwarg_dataflow_outputs(view);

    this->graph_input_labels = generate_map(
        view_inputs, [&](KwargDataflowGraphInput<GraphInputName> const &i) {
          return view.at(OpenKwargDataflowValue<GraphInputName, SlotName>{i});
        });

    this->nodes = view_nodes;
    this->node_labels =
        generate_map(view_nodes, [&](Node const &n) { return view.at(n); });

    for (OpenKwargDataflowEdge<GraphInputName, SlotName> const &e :
         view_edges) {
      this->add_edge(e);
    }

    for (KwargDataflowOutput<SlotName> const &o : view_outputs) {
      this->add_output(
          o, view.at(OpenKwargDataflowValue<GraphInputName, SlotName>{o}));
    }
  }

  UnorderedSetLabelledOpenKwargDataflowGraph *clone() const override {
    return new UnorderedSetLabelledOpenKwargDataflowGraph{
        this->node_source,
        this->nodes,
        this->internal_edges_index,
        this->external_edges_index,
        this->output_index,
        this->graph_input_labels,
        this->node_labels,
        this->output_labels,
    };
  }

private:
  UnorderedSetLabelledOpenKwargDataflowGraph(
      NodeSource const &node_source,
      std::set<Node> const &nodes,
      BiIndex<Node, Node, OpenKwargDataflowEdge<GraphInputName, SlotName>> const
          &internal_edges_index,
      BiIndex<GraphInputName,
              Node,
              OpenKwargDataflowEdge<GraphInputName, SlotName>> const
          &external_edges_index,
      BiIndex<Node, SlotName, KwargDataflowOutput<SlotName>> const
          &output_index,
      std::map<KwargDataflowGraphInput<GraphInputName>, ValueLabel> const
          &graph_input_labels,
      std::map<Node, NodeLabel> const &node_labels,
      std::map<KwargDataflowOutput<SlotName>, ValueLabel> const &output_labels)
      : node_source(node_source), internal_edges_index(internal_edges_index),
        external_edges_index(external_edges_index), output_index(output_index),
        graph_input_labels(graph_input_labels), node_labels(node_labels),
        output_labels(output_labels) {}

  void
      add_edge(OpenKwargDataflowEdge<GraphInputName, SlotName> const &in_edge) {
    if (in_edge.is_internal_edge()) {
      KwargDataflowEdge<SlotName> const &e = in_edge.require_internal_edge();

      this->internal_edges_index.add_value(e.src.node, e.dst.node, in_edge);
    } else {
      KwargDataflowInputEdge<GraphInputName, SlotName> const &e =
          in_edge.require_input_edge();

      this->external_edges_index.add_value(e.src.name, e.dst.node, in_edge);
    }
  }

  void add_output(KwargDataflowOutput<SlotName> const &o,
                  ValueLabel const &label) {
    this->output_labels.insert({o, label});
    this->output_index.add_value(o.node, o.slot_name, o);
  }

private:
  NodeSource node_source;

  std::set<Node> nodes;

  BiIndex<Node, Node, OpenKwargDataflowEdge<GraphInputName, SlotName>>
      internal_edges_index;
  BiIndex<GraphInputName, Node, OpenKwargDataflowEdge<GraphInputName, SlotName>>
      external_edges_index;
  BiIndex<Node, SlotName, KwargDataflowOutput<SlotName>> output_index;

  std::map<KwargDataflowGraphInput<GraphInputName>, ValueLabel>
      graph_input_labels;
  std::map<Node, NodeLabel> node_labels;
  std::map<KwargDataflowOutput<SlotName>, ValueLabel> output_labels;
};

} // namespace FlexFlow

#endif
