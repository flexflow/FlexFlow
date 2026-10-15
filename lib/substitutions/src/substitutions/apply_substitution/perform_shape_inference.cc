#include "substitutions/apply_substitution/perform_shape_inference.h"
#include "op-attrs/get_incoming_tensor_roles.h"
#include "op-attrs/shape_inference.h"
#include "utils/containers/binary_merge_disjoint_maps.h"
#include "utils/containers/filter_values.h"
#include "utils/containers/filtrans.h"
#include "utils/containers/is_subseteq_of.h"
#include "utils/containers/map_keys.h"
#include "utils/containers/map_values.h"
#include "utils/containers/restrict_keys_strict.h"
#include "utils/containers/transform.h"
#include "utils/containers/values.h"
#include "utils/containers/zip.h"
#include "utils/containers/zip_values_strict.h"
#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_topological_ordering.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_outgoing_kwarg_dataflow_outputs_for_node.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/rewrite_value_labels.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/rewrite_labelled_open_kwarg_dataflow_graph_value_labels.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_inputs.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_incoming_open_kwarg_dataflow_values_for_node.h"
#include "utils/nonnegative_int/num_elements.h"

namespace FlexFlow {

LabelledOpenKwargDataflowGraphView<ParallelLayerAttrs,
                                   ParallelTensorShape,
                                   int,
                                   TensorSlotName>
    perform_shape_inference(
        LabelledOpenKwargDataflowGraphView<ParallelLayerAttrs,
                                           std::monostate,
                                           int,
                                           TensorSlotName> const &g,
        std::map<KwargDataflowGraphInput<int>, ParallelTensorShape> const
            &input_shapes) {

  std::map<OpenKwargDataflowValue<int, TensorSlotName>, ParallelTensorShape>
      inferred =
          map_keys(input_shapes,
                   [](KwargDataflowGraphInput<int> const &i)
                       -> OpenKwargDataflowValue<int, TensorSlotName> {
                     return OpenKwargDataflowValue<int, TensorSlotName>{i};
                   });

  for (Node const &n : get_topological_ordering(g)) {
    std::map<TensorSlotName, ParallelTensorShape> incoming_shapes =
        map_values(get_incoming_open_kwarg_dataflow_values_for_node(g, n),
                   [&](OpenKwargDataflowValue<int, TensorSlotName> const &v) {
                     return inferred.at(v);
                   });

    ParallelLayerAttrs n_attrs = g.at(n);

    std::map<TensorSlotName, IncomingTensorRole> incoming_tensor_roles =
        get_incoming_tensor_roles(n_attrs.op_attrs);

    ASSERT(is_subseteq_of(keys(incoming_shapes), keys(incoming_tensor_roles)));

    auto incoming_shapes_with_role = [&](IncomingTensorRole role)
        -> std::map<TensorSlotName, ParallelTensorShape> {
      std::set<TensorSlotName> slots_with_desired_role =
          keys(filter_values(incoming_tensor_roles,
                             [&](IncomingTensorRole r) { return r == role; }));

      return restrict_keys_strict(incoming_shapes, slots_with_desired_role);
    };

    std::map<TensorSlotName, ParallelTensorShape> input_shapes =
        incoming_shapes_with_role(IncomingTensorRole::INPUT);
    std::map<TensorSlotName, ParallelTensorShape> weight_shapes =
        incoming_shapes_with_role(IncomingTensorRole::WEIGHT);

    ASSERT(binary_merge_disjoint_maps(input_shapes, weight_shapes) ==
           incoming_shapes);

    std::map<TensorSlotName, ParallelTensorShape> inferred_weight_shapes =
        get_weight_shapes(n_attrs.op_attrs, input_shapes);

    ASSERT(weight_shapes == inferred_weight_shapes);

    std::map<TensorSlotName, ParallelTensorShape> output_shapes =
        get_output_shapes(n_attrs.op_attrs, input_shapes);

    std::map<TensorSlotName, KwargDataflowOutput<TensorSlotName>> outputs =
        get_outgoing_kwarg_dataflow_outputs_for_node(g, n);

    for (auto const &[output, shape] :
         values(zip_values_strict(outputs, output_shapes))) {
      inferred.insert(
          {OpenKwargDataflowValue<int, TensorSlotName>{output}, shape});
    }
  }

  return rewrite_labelled_open_kwarg_dataflow_graph_value_labels(
      g,
      [&](OpenKwargDataflowValue<int, TensorSlotName> const &v,
          std::monostate const &) { return inferred.at(v); });
}

} // namespace FlexFlow
