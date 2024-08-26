#include "substitutions/substitution_internal/evaluate_substitution_output.h"
#include "substitutions/output_graph/output_operator_attrs_assignment.h"
#include "substitutions/sub_parallel_computation_graph.h"
#include "substitutions/substitution_internal/perform_shape_inference.h"
#include "utils/containers/map_keys.h"
#include "utils/containers/map_values.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/permute_input_ids.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/permute_node_ids.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/rewrite_node_labels.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/rewrite_value_labels.h"
#include "utils/graph/node/algorithms/generate_new_node_id_permutation.h"
#include "utils/graph/node/algorithms/new_node.dtg.h"
#include "utils/graph/open_dataflow_graph/algorithms/generate_new_input_id_permutation.h"
#include "utils/graph/open_dataflow_graph/algorithms/new_dataflow_graph_input.dtg.h"

namespace FlexFlow {

std::pair<SubParallelComputationGraph, OutputExprToResultSubPCGMapping>
    evaluate_substitution_output(SubParallelComputationGraph const &spcg,
                                 Substitution const &sub,
                                 PCGPatternMatch const &match) {
  std::unordered_map<PatternNode, PCGOperatorAttrs> node_match =
      map_values(match.node_assignment.as_unordered_map(),
                 [&](parallel_layer_guid_t const &n) {
                   return get_operator_attrs(spcg, n);
                 });

  bidict<NewNode, Node> new_node_id_permutation =
      generate_new_node_id_permutation(sub.output_graph_expr.raw_graph);
  bidict<NewDataflowGraphInput, DataflowGraphInput> new_input_id_permutation =
      generate_new_input_id_permutation(sub.output_graph_expr.raw_graph);
  LabelledOpenDataflowGraphView<OutputOperatorAttrsAssignment, std::monostate>
      permuted =
          permute_input_ids(permute_node_ids(sub.output_graph_expr.raw_graph,
                                             new_node_id_permutation),
                            new_input_id_permutation);

  LabelledOpenDataflowGraphView<ParallelLayerAttrs, std::monostate>
      without_shapes = rewrite_node_labels(
          permuted,
          [&](Node const &n, OutputOperatorAttrsAssignment const &attrs) {
            return ParallelLayerAttrs{
                materialize_output_operator_from_attrs_assignment(attrs,
                                                                  node_match),
                std::nullopt,
            };
          });

  bidict<input_parallel_tensor_guid_t, OutputGraphExprInput> result_input_map =
      map_keys(map_values(new_input_id_permutation,
                          [](DataflowGraphInput const &i) {
                            return OutputGraphExprInput{i};
                          }),
               [](NewDataflowGraphInput const &i) {
                 return input_parallel_tensor_guid_t{i.raw_input};
               });

  bidict<parallel_layer_guid_t, OutputGraphExprNode> result_node_map = map_keys(
      map_values(new_node_id_permutation,
                 [](Node const &n) { return OutputGraphExprNode{n}; }),
      [](NewNode const &n) { return parallel_layer_guid_t{n.raw_node}; });

  std::unordered_map<DataflowGraphInput, ParallelTensorShape> input_shapes =
      map_values(map_keys(match.input_assignment,
                          [&](PatternInput const &i) {
                            return result_input_map
                                .at_r(sub.inputs_mapping.at_l(i))
                                .raw_dataflow_graph_input;
                          }),
                 [&](open_parallel_tensor_guid_t const &v) {
                   return spcg.raw_graph.at(v.raw_open_dataflow_value).shape;
                 });
  LabelledOpenDataflowGraphView<ParallelLayerAttrs, ParallelTensorShape>
      with_shapes = perform_shape_inference(without_shapes, input_shapes);
  LabelledOpenDataflowGraphView<ParallelLayerAttrs, ParallelTensorAttrs>
      with_attrs = rewrite_value_labels(
          with_shapes,
          [](OpenDataflowValue const &, ParallelTensorShape const &s) {
            return ParallelTensorAttrs{
                s,
                std::nullopt,
                std::nullopt,
                CreateGrad::YES,
            };
          });

  return std::make_pair(SubParallelComputationGraph{with_attrs},
                        OutputExprToResultSubPCGMapping{
                            result_node_map,
                            result_input_map,
                        });
}

} // namespace FlexFlow
