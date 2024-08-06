#include "substitutions/substitution.h"
#include "substitutions/output_graph/output_operator_attrs_assignment.h"
#include "substitutions/sub_parallel_computation_graph.h"
#include "utils/bidict/algorithms/right_entries.h"
#include "utils/containers/set_minus.h"
#include "utils/containers/values.h"
#include "utils/containers/map_values.h"
#include "utils/graph/digraph/algorithms/get_topological_ordering.h"
#include "utils/graph/instances/unordered_set_labelled_open_dataflow_graph.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/permute_input_ids.h"
#include "utils/graph/node/algorithms/generate_new_node_id_permutation.h"
#include "utils/graph/open_dataflow_graph/algorithms/generate_new_input_id_permutation.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/rewrite_labels.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/rewrite_node_labels.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/rewrite_value_labels.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_open_dataflow_value_uses.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/permute_node_ids.h"
#include "utils/overload.h"
#include "utils/containers/transform.h"
#include "op-attrs/get_output_shapes.h"
#include "utils/graph/node/algorithms.h"

namespace FlexFlow {

/* struct AddMappedEdgeFunctor { */
/*   bidict<Node, Node> const &node_mapping; */
/*   SubParallelComputationGraph &new_pcg; */

/*   template <typename T> */
/*   void operator()(T const &t) { */
/*     return add_mapped_edge(t); */
/*   } */

/*   void add_mapped_edge(InputMultiDiEdge const &e) { */
/*     new_pcg.add_edge(InputMultiDiEdge{ */
/*         node_mapping.at_l(e.dst), new_pcg.add_node_port(), e.uid}); */
/*   } */

/*   void add_mapped_edge(OutputMultiDiEdge const &e) { */
/*     new_pcg.add_edge(OutputMultiDiEdge{ */
/*         node_mapping.at_l(e.src), new_pcg.add_node_port(), e.uid}); */
/*   } */

/*   void add_mapped_edge(MultiDiEdge const &e) { */
/*     new_pcg.add_edge(MultiDiEdge{node_mapping.at_l(e.dst), */
/*                                  new_pcg.add_node_port(), */
/*                                  node_mapping.at_l(e.src), */
/*                                  new_pcg.add_node_port()}); */
/*   } */
/* }; */

/* struct AddNewEdgeFunctor { */
/*   SubParallelComputationGraph const &old_pcg; */
/*   SubParallelComputationGraph &new_pcg; */
/*   MultiDiGraphPatternMatch const &match; */
/*   bidict<Node, Node> node_mapping; */

/*   template <typename TO, typename TN> */
/*   void operator()(TO const &old_edge, TN const &new_edge) { */
/*     return add_new_edge(old_edge, new_edge); */
/*   } */

/*   void add_new_edge(InputMultiDiEdge const &old_edge, */
/*                     InputMultiDiEdge const &new_edge) { */
/*     new_pcg.add_edge(InputMultiDiEdge{node_mapping.at_l(new_edge.dst), */
/*                                       new_pcg.add_node_port(), */
/*                                       old_edge.uid}); */
/*   } */

/*   void add_new_edge(MultiDiEdge const &old_edge, */
/*                     InputMultiDiEdge const &new_edge) { */
/*     new_pcg.add_edge(MultiDiEdge{node_mapping.at_l(new_edge.dst), */
/*                                  new_pcg.add_node_port(), */
/*                                  node_mapping.at_l(old_edge.src), */
/*                                  new_pcg.add_node_port()}); */
/*   } */

/*   void add_new_edge(OutputMultiDiEdge const &old_edge, */
/*                     OutputMultiDiEdge const &new_edge) { */
/*     new_pcg.add_edge(OutputMultiDiEdge{node_mapping.at_l(new_edge.src), */
/*                                        new_pcg.add_node_port(), */
/*                                        old_edge.uid}); */
/*   } */

/*   void add_new_edge(MultiDiEdge const &old_edge, */
/*                     OutputMultiDiEdge const &new_edge) { */
/*     new_pcg.add_edge(MultiDiEdge{node_mapping.at_l(old_edge.dst), */
/*                                  new_pcg.add_node_port(), */
/*                                  node_mapping.at_l(new_edge.src), */
/*                                  new_pcg.add_node_port()}); */
/*   } */

/*   void add_new_edge(InputMultiDiEdge const &, OutputMultiDiEdge const &) { */
/*     assert(false); */
/*   } */

/*   void add_new_edge(OpenMultiDiEdge const &, MultiDiEdge const &) { */
/*     assert(false); */
/*   } */

/*   void add_new_edge(OutputMultiDiEdge const &, InputMultiDiEdge const &) { */
/*     assert(false); */
/*   } */
/* }; */

/* SubParallelComputationGraph */
/*     apply_substitution(SubParallelComputationGraph const &pcg, */
/*                        Substitution const &substitution, */
/*                        MultiDiGraphPatternMatch const &match) { */
/*   SubParallelComputationGraph new_pcg = */
/*       OutputLabelledOpenMultiDiGraph<Operator, ParallelTensor>::template
 * create< */
/*           UnorderedOutputLabelledOpenMultiDiGraph<Operator,
 * ParallelTensor>>(); */
/*   bidict<Node, Node> node_mapping; // Refactor it with global nodes */
/*   for (Node const &node : get_nodes(pcg)) { */
/*     if (!contains_r(match.node_assignment, node)) { */
/*       node_mapping.equate(node, new_pcg.add_node(pcg.at(node))); */
/*     } */
/*   } */
/*   for (OpenMultiDiEdge const &edge : get_edges(pcg)) { */
/*     if (!contains_r(match.edge_assignment, edge)) { */
/*       visit(AddMappedEdgeFunctor{node_mapping, new_pcg}, edge); */
/*     } */
/*   } */
/*   for (Node const &output_node : */
/*        get_nodes(substitution.output_graph_expr.value())) { */
/*     Operator new_op = get_operator_attrs( */
/*         pcg, match, substitution.output_graph_expr.value().at(output_node));
 */
/*     Node new_node = new_pcg.add_node(new_op); */
/*     node_mapping.equate(output_node, new_node); */
/*   } */
/*   for (OpenMultiDiEdge const &output_edge : */
/*        get_edges(substitution.output_graph_expr.value())) { */
/*     if (std::holds_alternative<InputMultiDiEdge>(output_edge)) { */
/*       InputMultiDiEdge e = std::get<InputMultiDiEdge>(output_edge); */
/*       OpenMultiDiEdge original_edge = */
/*           match.edge_assignment.at_l(substitution.input_mapping.at_r(e)); */
/*       visit(AddNewEdgeFunctor{pcg, new_pcg, match, node_mapping}, */
/*             original_edge, */
/*             output_edge); */
/*     } else if (std::holds_alternative<OutputMultiDiEdge>(output_edge)) { */
/*       OutputMultiDiEdge e = std::get<OutputMultiDiEdge>(output_edge); */
/*       OpenMultiDiEdge original_edge = */
/*           match.edge_assignment.at_l(substitution.output_mapping.at_r(e)); */
/*       visit(AddNewEdgeFunctor{pcg, new_pcg, match, node_mapping}, */
/*             original_edge, */
/*             output_edge); */
/*     } else { */
/*       assert(std::holds_alternative<MultiDiEdge>(output_edge)); */
/*       MultiDiEdge e = std::get<MultiDiEdge>(output_edge); */
/*       new_pcg.add_edge(MultiDiEdge{node_mapping.at_l(e.dst), */
/*                                    new_pcg.add_node_port(), */
/*                                    node_mapping.at_l(e.src), */
/*                                    new_pcg.add_node_port()}); */
/*     } */
/*   } */

/*   return new_pcg; */
/* } */

bool is_valid_substitution(Substitution const &) {
  NOT_IMPLEMENTED();
}

LabelledOpenDataflowGraphView<ParallelLayerAttrs, ParallelTensorShape> perform_shape_inference(
  LabelledOpenDataflowGraphView<ParallelLayerAttrs, std::monostate> const &g,
  std::unordered_map<DataflowGraphInput, ParallelTensorShape> const &input_shapes) {
  
  std::unordered_map<OpenDataflowValue, ParallelTensorShape> inferred = map_keys(input_shapes, 
                                                                                 [](DataflowGraphInput const &i) { return OpenDataflowValue{i}; });

  for (Node const &n : get_topological_ordering(g)) {
    std::vector<ParallelTensorShape> input_shapes = transform(get_inputs(g, n), [&](OpenDataflowValue const &v) { return inferred.at(v); });
    std::vector<ParallelTensorShape> output_shapes = get_output_shapes(g.at(n).op_attrs, input_shapes);
    std::vector<DataflowOutput> outputs = get_outputs(g, n);

    for (auto const &[output, shape] : zip(outputs, output_shapes)) {
      inferred.insert({OpenDataflowValue{output}, shape});
    }
  }

  return rewrite_value_labels(g, [&](OpenDataflowValue const &v, std::monostate const &) { return inferred.at(v); });
}

std::pair<
  SubParallelComputationGraph,
  OutputExprToResultSubPCGMapping
> evaluate_substitution_output(SubParallelComputationGraph const &spcg,
                               Substitution const &sub,
                               PCGPatternMatch const &match) {
  std::unordered_map<PatternNode, PCGOperatorAttrs> node_match = map_values(match.node_assignment.as_unordered_map(), 
                                                                            [&](parallel_layer_guid_t const &n) { return get_operator_attrs(spcg, n); });

  bidict<NewNode, Node> new_node_id_permutation = generate_new_node_id_permutation(sub.output_graph_expr.raw_graph);
  bidict<NewDataflowGraphInput, DataflowGraphInput> new_input_id_permutation = generate_new_input_id_permutation(sub.output_graph_expr.raw_graph);
  LabelledOpenDataflowGraphView<OutputOperatorAttrsAssignment, std::monostate> permuted = permute_input_ids(permute_node_ids(
                                                                                                                             sub.output_graph_expr.raw_graph,
                                                                                                                             new_node_id_permutation
                                                                                                                            ),
                                                                                                            new_input_id_permutation
                                                                                                           );

  LabelledOpenDataflowGraphView<ParallelLayerAttrs, std::monostate> without_shapes = 
    rewrite_node_labels(permuted,
      [&](Node const &n, OutputOperatorAttrsAssignment const &attrs) { 
        return ParallelLayerAttrs{
          materialize_output_operator_from_attrs_assignment(attrs, node_match),
          std::nullopt,
        };
      }
    );

  bidict<input_parallel_tensor_guid_t, OutputGraphExprInput> result_input_map = map_keys(map_values(new_input_id_permutation,
                                                                                                    [](DataflowGraphInput const &i) { return OutputGraphExprInput{i}; }),
                                                                                         [](NewDataflowGraphInput const &i) { return input_parallel_tensor_guid_t{i.raw_input}; });

  bidict<parallel_layer_guid_t, OutputGraphExprNode> result_node_map = map_keys(map_values(new_node_id_permutation,
                                                                                           [](Node const &n) { return OutputGraphExprNode{n}; }),
                                                                                [](NewNode const &n) { return parallel_layer_guid_t{n.raw_node}; });


  std::unordered_map<DataflowGraphInput, ParallelTensorShape> input_shapes = map_values(map_keys(match.input_assignment,
                                                                                      [&](PatternInput const &i) { return result_input_map.at_r(sub.inputs_mapping.at_l(i)).raw_dataflow_graph_input; }),
                                                                                      [&](open_parallel_tensor_guid_t const &v) { return spcg.raw_graph.at(v.raw_open_dataflow_value).shape; });
  LabelledOpenDataflowGraphView<ParallelLayerAttrs, ParallelTensorShape> with_shapes = perform_shape_inference(without_shapes, input_shapes);
  LabelledOpenDataflowGraphView<ParallelLayerAttrs, ParallelTensorAttrs> with_attrs = rewrite_value_labels(with_shapes, 
                                                                                                           [](OpenDataflowValue const &, ParallelTensorShape const &s) {
                                                                                                             return ParallelTensorAttrs{
                                                                                                               s, 
                                                                                                               std::nullopt,
                                                                                                               std::nullopt,
                                                                                                               CreateGrad::YES,
                                                                                                             };
                                                                                                            });

  return std::make_pair(
   SubParallelComputationGraph{with_attrs},
   OutputExprToResultSubPCGMapping{
     result_node_map,
     result_input_map,
    }
  );
}


struct SubstitutionAppliedView final : ILabelledOpenDataflowGraphView<ParallelLayerAttrs, ParallelTensorAttrs> {
  SubstitutionAppliedView(Substitution const &sub,
                          PCGPatternMatch const &match,
                          SubParallelComputationGraph const &spcg,
                          SubParallelComputationGraph const &substitution_result,
                          OutputExprToResultSubPCGMapping const &output_expr_to_result_sub_pcg_mapping)
    : sub(sub),
      match(match),
      spcg(spcg),
      substitution_result(substitution_result),
      output_expr_to_result_sub_pcg_mapping(output_expr_to_result_sub_pcg_mapping)
    { }


  std::unordered_set<Node> query_nodes(NodeQuery const &q) const override {
    std::unordered_set<Node> original_result = this->spcg.raw_graph.query_nodes(q);
    std::unordered_set<Node> matched_by_subsitution = transform(right_entries(this->match.node_assignment), [](parallel_layer_guid_t const &l) { return l.raw_graph_node; });;
    std::unordered_set<Node> substitution_output_result = this->substitution_result.raw_graph.query_nodes(q);
    return set_minus(set_union(original_result, substitution_output_result), matched_by_subsitution);
  }

  std::unordered_set<DataflowGraphInput> get_inputs() const override {
    return this->spcg.raw_graph.get_inputs();
  }

  std::unordered_set<DataflowOutput> query_outputs(DataflowOutputQuery const &q) const override {
    NOT_IMPLEMENTED();
    // return this->spcg.query_outputs(q);
  }

  std::unordered_set<OpenDataflowEdge> query_edges(OpenDataflowEdgeQuery const &q) const override {
    std::unordered_set<OpenDataflowEdge> original_result = this->spcg.raw_graph.query_edges(q);
    std::unordered_set<OpenDataflowEdge> substitution_output_result = this->substitution_result.raw_graph.query_edges(q);

    std::unordered_set<OpenDataflowEdge> from_original = filter(original_result, 
                                                                [&](OpenDataflowEdge const &e) {
                                                                  if (e.has<DataflowInputEdge>()) {
                                                                    return true;
                                                                  } else {
                                                                    DataflowEdge dfe = e.get<DataflowEdge>();
                                                                    if (has_node(this->substitution_result.raw_graph, dfe.src.node)
                                                                        || this->match.node_assignment.contains_r(parallel_layer_guid_t{dfe.src.node})) {
                                                                      return false;
                                                                    }
                                                                    if (has_node(this->substitution_result.raw_graph, dfe.dst.node)
                                                                        || this->match.node_assignment.contains_r(parallel_layer_guid_t{dfe.dst.node})) {
                                                                      return false;
                                                                    }

                                                                    return true;
                                                                  }
                                                                });

    std::unordered_set<OpenDataflowEdge> from_output = filter(substitution_output_result,
                                                              [&](OpenDataflowEdge const &e) {
                                                                return !e.has<DataflowInputEdge>();
                                                              });

    std::unordered_set<OpenDataflowEdge> incoming_to_output;
    for (auto const &[pattern_input, base_graph_value] : this->match.input_assignment) {
      OutputGraphExprInput output_expr_input = this->sub.inputs_mapping.at_l(pattern_input);
      for (DataflowInput const &use : get_open_dataflow_value_uses(this->substitution_result.raw_graph, OpenDataflowValue{pattern_input.raw_dataflow_graph_input})) {
        incoming_to_output.insert(
          open_dataflow_edge_from_src_and_dst(base_graph_value.raw_open_dataflow_value, use));
      }
    }

    NOT_IMPLEMENTED();

    // std::unordered_set<OpenDataflowEdge> outgoing_from_output;
    // for (auto const &[pattern_output, output_graph_output] : this->sub.

    // std::unordered_set<OpenDataflowEdge> outgoing_from_output = filter(original_result, 
    //                                                                    [&](OpenDataflowEdge const &e) {
    //                                                                      if (e.has<DataflowInputEdge>()) {
    //                                                                        return false;
    //                                                                      }
    //
    //                                                                      DataflowEdge dfe = e.get<DataflowEdge>();
    //                                                                      if (!has_node(this->substitution_result.raw_graph, dfe.src.node)) {
    //                                                                        return false;
    //                                                                      }
    //                                                                      if (has_node(this->substitution_result.raw_graph, dfe.dst.node)
    //                                                                          || this->match.node_assignment.contains_r(dfe.dst.node)) {
    //                                                                        return false;
    //                                                                      }
    //
    //                                                                      return OpenDataflowEdge{
    //                                                                        DataflowEdge{
    //                                                                          this->sub.outputs_mapping.at_
    //                                                                          dfe.dst,
    //                                                                        },
    //                                                                      }
    //                                                                    });
    // std::unordered_set<OpenDataflowEdge> incoming_to_output = filter(original_result,
    //                                                                  [&](OpenDataflowEdge const &e) {
    //                                                                    if (e.has<DataflowInputEdge>()) {
    //                                                                      return false;
    //                                                                    }
    //
    //                                                                    DataflowEdge dfe = e.get<DataflowEdge>();
    //                                                                    if (has_node(this->substitution_result.raw_graph, dfe.src.node)
    //                                                                        || this->match.node_assignment.contains_r(dfe.src.node)) {
    //                                                                      return false;
    //                                                                    }
    //                                                                    if (!has_node(this->substitution_result.raw_graph, dfe.dst.node)) {
    //                                                                      return false;
    //                                                                    }
    //
    //                                                                    return OpenDataflowValue{
    //                                                                      DataflowEdge{
    //                                                                        dfe.src,
    //                                                                        dfe.dst,
    //                                                                      }
    //                                                                    };
    //                                                                  });
  }


  ParallelLayerAttrs at(Node const &n) const override {
    if (has_node(this->substitution_result.raw_graph, n)) {
      return this->substitution_result.raw_graph.at(n);
    } else if (this->match.node_assignment.contains_r(parallel_layer_guid_t{n})) {
      throw mk_runtime_error("Cannot access node that has been replaced out by a substitution");
    } else {
      return this->spcg.raw_graph.at(n);
    }
  }

  ParallelTensorAttrs at(OpenDataflowValue const &v) const override {
    return v.visit<ParallelTensorAttrs>(overload {
      [&](DataflowGraphInput const &i) { return this->spcg.raw_graph.at(OpenDataflowValue{i}); },
      [&](DataflowOutput const &o) { 
        if (has_node(this->substitution_result.raw_graph, o.node)) {
          return this->substitution_result.raw_graph.at(OpenDataflowValue{o});
        } else if (this->match.node_assignment.contains_r(parallel_layer_guid_t{o.node})) {
          throw mk_runtime_error("Cannot access node that has been replaced out by a substitution");
        } else {
          return this->spcg.raw_graph.at(OpenDataflowValue{o});
        }
      },
    });
  }

  SubstitutionAppliedView *clone() const override {
    return new SubstitutionAppliedView(
      this->sub,
      this->match,
      this->spcg,
      this->substitution_result,
      this->output_expr_to_result_sub_pcg_mapping
    );
  }
private:
  Substitution sub;
  PCGPatternMatch match;
  SubParallelComputationGraph spcg;
  SubParallelComputationGraph substitution_result;
  OutputExprToResultSubPCGMapping output_expr_to_result_sub_pcg_mapping;
};

SubParallelComputationGraph
    apply_substitution(SubParallelComputationGraph const &spcg,
                       Substitution const &sub,
                       PCGPatternMatch const &match) {
  auto substitution_output_result = evaluate_substitution_output(spcg, sub, match);
  SubParallelComputationGraph substitution_output_graph = substitution_output_result.first;
  OutputExprToResultSubPCGMapping output_expr_to_result_sub_pcg_mapping = substitution_output_result.second;

  return SubParallelComputationGraph{
    LabelledOpenDataflowGraphView<ParallelLayerAttrs, ParallelTensorAttrs>::create<SubstitutionAppliedView>(
      sub,
      match,
      spcg,
      substitution_output_graph,
      output_expr_to_result_sub_pcg_mapping
    )
  };
}

} // namespace FlexFlow
