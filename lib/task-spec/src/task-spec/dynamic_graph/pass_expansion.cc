#include "task-spec/dynamic_graph/pass_expansion.h"
#include "op-attrs/tensor_slot_name.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_tensor_role.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.h"
#include "task-spec/dynamic_graph/training_operation_attrs.h"
#include "utils/containers/any_of.h"
#include "utils/containers/binary_merge_disjoint_maps.h"
#include "utils/containers/filter.h"
#include "utils/containers/flatmap.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/get_only.h"
#include "utils/containers/invert_map.h"
#include "utils/containers/map_from_keys_and_values.h"
#include "utils/containers/map_from_pairs.h"
#include "utils/containers/map_values.h"
#include "utils/containers/merge_disjoint_maps.h"
#include "utils/containers/range.h"
#include "utils/containers/repeat_until_converged.h"
#include "utils/containers/set_intersection.h"
#include "utils/containers/set_of.h"
#include "utils/containers/slice.h"
#include "utils/containers/transform.h"
#include "utils/containers/try_at.h"

namespace FlexFlow {

void require_node_might_be_pass_expanded(DynamicNodeAttrs const &n) {
  if (assert_unwrap(n.op_attrs).is_copy()) {
    return;
  }

  ASSERT(n.task_type.has_value(), n);
}

void require_node_might_not_be_pass_expanded(DynamicNodeAttrs const &n) {
  if (assert_unwrap(n.op_attrs).is_copy()) {
    return;
  }

  ASSERT(!n.task_type.has_value(), n);
  ASSERT(!assert_unwrap(n.op_attrs).is_gradient_reduction());
}

void require_slot_is_not_pass_expanded(DynamicTensorSlot const &s) {
  ASSERT(!s.slot_tensor_role.has_value(), s);
}

void require_value_is_pass_expanded(DynamicValueAttrs const &v) {
  ASSERT(v.role.has_value(), v);
}

void require_value_is_not_pass_expanded(DynamicValueAttrs const &v) {
  ASSERT(!v.role.has_value(), v);
  ASSERT(!v.subgradient_id.has_value(), v);
}

void require_invocation_is_fully_pass_expanded(
    DynamicNodeInvocation const &invocation) {
  auto require_slot_is_pass_expanded = [&](DynamicTensorSlot const &s) {
    if (dynamic_node_invocation_get_op_type(invocation) ==
        TrainingOpType{TrainingOnlyOpType::COPY}) {
      return;
    }

    ASSERT(s.slot_tensor_role.has_value(), s);
  };

  require_invocation_fully_satisfies(invocation,
                                     require_node_might_be_pass_expanded,
                                     require_value_is_pass_expanded,
                                     require_slot_is_pass_expanded);
}

void require_invocation_is_ready_for_pass_expansion(
    DynamicNodeInvocation const &invocation) {
  require_invocation_fully_satisfies(invocation,
                                     require_node_might_not_be_pass_expanded,
                                     require_value_is_not_pass_expanded,
                                     require_slot_is_not_pass_expanded);
}

void require_graph_is_fully_pass_expanded(DynamicOpenDataflowGraph const &g) {
  require_full_dynamic_graph_satisfies(
      g, require_invocation_is_fully_pass_expanded);
}

void require_graph_is_ready_for_pass_expansion(
    DynamicOpenDataflowGraph const &g) {
  require_full_dynamic_graph_satisfies(
      g, require_invocation_is_ready_for_pass_expansion);
}

std::set<DynamicValueAttrs>
    determine_intermediate_values_needed_to_compute_gradients_of_value(
        DynamicOpenDataflowGraph const &g, DynamicValueAttrs const &val) {
  auto get_values_immediately_needed_to_compute_gradients_of_needed =
      [&](std::set<DynamicValueAttrs> const &needed)
      -> std::set<DynamicValueAttrs> {
    std::set<DynamicValueAttrs> additional = flatmap(
        needed, [&](DynamicValueAttrs const &v) -> std::set<DynamicValueAttrs> {
          std::set<InternalDynamicSlotSite> sinks =
              dynamic_graph_find_sinks_of_value(g, v);

          return flatmap(sinks,
                         [&](InternalDynamicSlotSite const &sink)
                             -> std::set<DynamicValueAttrs> {
                           DynamicNodeInvocation sink_invocation =
                               dynamic_graph_get_invocation_for_id(
                                   g, sink.invocation_id);

                           return set_of(values(sink_invocation.outputs));
                         });
        });

    return set_union(needed, additional);
  };

  std::set<DynamicValueAttrs> result = repeat_until_converged(
      std::set<DynamicValueAttrs>{val},
      get_values_immediately_needed_to_compute_gradients_of_needed);

  ASSERT(contains(result, val));
  return result;
}

std::set<DynamicValueAttrs>
    determine_intermediate_values_needed_for_gradient_computation(
        DynamicOpenDataflowGraph const &g) {
  auto value_is_fundamentally_required =
      [&](DynamicValueAttrs const &v) -> bool {
    DynamicSlotSite source = dynamic_graph_find_source_of_value(g, v);

    if (source.is_external()) {
      return true;
    }

    InternalDynamicSlotSite internal_source = source.require_internal();
    ASSERT(internal_source.direction == TensorDirection::OUTPUT);

    DynamicNodeInvocation source_invocation =
        dynamic_graph_get_invocation_for_id(g, internal_source.invocation_id);

    TrainingOpType op_type =
        dynamic_node_invocation_get_op_type(source_invocation);

    TrainingOpType weight_op_type = TrainingOpType{OperatorType::WEIGHT};
    TrainingOpType input_op_type = TrainingOpType{OperatorType::INPUT};

    if (op_type == weight_op_type) {
      return true;
    } else if (op_type == input_op_type) {
      return assert_unwrap(v.create_grad);
    } else {
      return false;
    }
  };

  std::set<DynamicValueAttrs> fundamentally_required_values =
      filter(set_of(get_dynamic_values(g)), value_is_fundamentally_required);

  std::set<DynamicValueAttrs> required_values = flatmap(
      fundamentally_required_values,
      [&](DynamicValueAttrs const &fundamentally_required_value)
          -> std::set<DynamicValueAttrs> {
        return determine_intermediate_values_needed_to_compute_gradients_of_value(
            g, fundamentally_required_value);
      });

  return required_values;
}

std::set<dynamic_invocation_id_t>
    determine_invocations_needed_in_backward_pass_for_gradient_computation(
        DynamicOpenDataflowGraph const &g) {
  std::set<DynamicValueAttrs> required_values =
      determine_intermediate_values_needed_for_gradient_computation(g);

  auto get_sink_invocations_for_value =
      [&](DynamicValueAttrs const &v) -> std::set<dynamic_invocation_id_t> {
    return transform(dynamic_graph_find_sinks_of_value(g, v),
                     [&](InternalDynamicSlotSite const &sink_site)
                         -> dynamic_invocation_id_t {
                       ASSERT(sink_site.direction == TensorDirection::INCOMING);
                       return sink_site.invocation_id;
                     });
  };

  return flatmap(required_values, get_sink_invocations_for_value);
}

std::set<DynamicValueAttrs> determine_values_requiring_gradient_reduction(
    DynamicOpenDataflowGraph const &g,
    std::set<dynamic_invocation_id_t> const &in_bwd_pass) {
  std::set<DynamicValueAttrs> internal_values =
      dynamic_graph_get_internal_values(g);
  return filter(internal_values, [&](DynamicValueAttrs const &v) {
    std::set<InternalDynamicSlotSite> sinks =
        dynamic_graph_find_sinks_of_value(g, v);
    if (sinks.size() <= 1) {
      return false;
    }
    return any_of(sinks, [&](InternalDynamicSlotSite const &sink_site) {
      return in_bwd_pass.count(sink_site.invocation_id);
    });
  });
}

static std::map<dynamic_invocation_id_t, subgradient_id_t>
    assign_subgradient_ids(
        std::set<dynamic_invocation_id_t> const &invocations) {
  const std::vector<TensorSlotName> slot_names =
      get_variadic_inputs_slot_name_sequence();
  ASSERT(invocations.size() <= slot_names.size());
  return map_from_keys_and_values(
      vector_of(invocations),
      transform(slice(slot_names, 0, invocations.size()),
                [](TensorSlotName const &slot_name) {
                  return subgradient_id_t{slot_name};
                }));
}

std::map<DynamicValueAttrs, std::map<dynamic_invocation_id_t, subgradient_id_t>>
    compute_subgradient_ids(DynamicOpenDataflowGraph const &g,
                            std::set<DynamicValueAttrs> const
                                &values_requiring_gradient_reduction) {
  return generate_map(
      values_requiring_gradient_reduction, [&](DynamicValueAttrs const &v) {
        return assign_subgradient_ids(
            transform(dynamic_graph_find_sinks_of_value(g, v),
                      [](InternalDynamicSlotSite const &sink_site) {
                        return sink_site.invocation_id;
                      }));
      });
}

DynamicTensorSlot pass_expand_slot(DynamicTensorSlot const &s,
                                   FwbTensorType tensor_type) {
  require_slot_is_not_pass_expanded(s);

  DynamicTensorSlot result = s;
  result.slot_tensor_role =
      dynamic_tensor_role_from_fwb_tensor_type(tensor_type);
  return result;
}

DynamicValueAttrs pass_expand_value(DynamicValueAttrs const &v,
                                    FwbTensorType tensor_type) {
  require_value_is_not_pass_expanded(v);

  DynamicValueAttrs result = v;
  result.role = DynamicTensorRole{tensor_type};
  return result;
};

DynamicNodeAttrs pass_expand_node(DynamicNodeAttrs const &n,
                                  DynamicTaskType task_type) {
  require_node_might_not_be_pass_expanded(n);

  ASSERT(task_type == DynamicTaskType::FWD ||
         task_type == DynamicTaskType::BWD);

  {
    TrainingOperationAttrs op_attrs = assert_unwrap(n.op_attrs);

    if (op_attrs.is_copy()) {
      return n;
    }
  }

  DynamicNodeAttrs result = n;
  result.task_type = task_type;
  return result;
}

DynamicNodeInvocation perform_fwd_pass_expansion_for_invocation(
    DynamicNodeInvocation const &invocation) {

  require_invocation_is_ready_for_pass_expansion(invocation);

  TrainingOperationAttrs op_attrs =
      assert_unwrap(invocation.node_attrs.op_attrs);

  auto to_fwd_value = [](DynamicValueAttrs const &v) -> DynamicValueAttrs {
    return pass_expand_value(v, FwbTensorType::FORWARD);
  };

  auto to_fwd = [](DynamicTensorSlot const &k, DynamicValueAttrs const &v) {
    return std::pair{
        pass_expand_slot(k, FwbTensorType::FORWARD),
        pass_expand_value(v, FwbTensorType::FORWARD),
    };
  };

  DynamicNodeInvocation result = [&]() -> DynamicNodeInvocation {
    if (op_attrs.is_copy()) {
      return DynamicNodeInvocation{
          /*inputs=*/map_values(invocation.inputs, to_fwd_value),
          /*node_attrs=*/invocation.node_attrs,
          /*outputs=*/map_values(invocation.outputs, to_fwd_value),
      };
    } else {
      return DynamicNodeInvocation{
          /*inputs=*/
          transform(invocation.inputs, to_fwd),
          /*node_attrs=*/
          pass_expand_node(invocation.node_attrs, DynamicTaskType::FWD),
          /*outputs=*/
          transform(invocation.outputs, to_fwd),
      };
    }
  }();

  require_invocation_is_fully_pass_expanded(result);

  return result;
}

DynamicNodeInvocation perform_bwd_pass_expansion_for_invocation(
    dynamic_invocation_id_t const &invocation_id,
    DynamicNodeInvocation const &invocation,
    std::map<DynamicValueAttrs,
             std::map<dynamic_invocation_id_t, subgradient_id_t>> const
        &subgradient_ids_by_value) {

  require_invocation_is_ready_for_pass_expansion(invocation);

  TrainingOperationAttrs op_attrs =
      assert_unwrap(invocation.node_attrs.op_attrs);

  auto subgradient_id_for_value = [&](DynamicValueAttrs const &v) {
    return transform(
        try_at(subgradient_ids_by_value, v),
        [&](std::map<dynamic_invocation_id_t, subgradient_id_t> const &s) {
          return s.at(invocation_id);
        });
  };

  auto pass_expand_value_with_subgradient = [&](DynamicValueAttrs const &v,
                                                FwbTensorType tensor_type) {
    std::optional<subgradient_id_t> subgradient_id =
        subgradient_id_for_value(v);
    DynamicValueAttrs result = pass_expand_value(v, tensor_type);
    if (subgradient_id.has_value()) {
      result = decide_dynamic_value_attrs_subgradient_id(
          result, assert_unwrap(subgradient_id));
    }
    return result;
  };

  auto to_fwd = [](DynamicTensorSlot const &k, DynamicValueAttrs const &v) {
    return std::pair{
        pass_expand_slot(k, FwbTensorType::FORWARD),
        pass_expand_value(v, FwbTensorType::FORWARD),
    };
  };

  auto to_grad_value = [](DynamicValueAttrs const &v) {
    return pass_expand_value(v, FwbTensorType::GRADIENT);
  };

  auto to_grad_value_output = [&](DynamicValueAttrs const &v) {
    return pass_expand_value_with_subgradient(v, FwbTensorType::GRADIENT);
  };

  auto to_grad = [](DynamicTensorSlot const &k, DynamicValueAttrs const &v) {
    return std::pair{
        pass_expand_slot(k, FwbTensorType::GRADIENT),
        pass_expand_value(v, FwbTensorType::GRADIENT),
    };
  };

  auto to_grad_output = [&](DynamicTensorSlot const &k,
                            DynamicValueAttrs const &v) {
    return std::pair{
        pass_expand_slot(k, FwbTensorType::GRADIENT),
        pass_expand_value_with_subgradient(v, FwbTensorType::GRADIENT),
    };
  };

  DynamicNodeInvocation result = [&]() -> DynamicNodeInvocation {
    if (op_attrs.is_copy()) {
      return DynamicNodeInvocation{
          /*inputs=*/map_values(invocation.outputs, to_grad_value),
          /*node_attrs=*/invocation.node_attrs,
          /*outputs=*/map_values(invocation.inputs, to_grad_value_output),
      };
    } else if (training_op_attrs_has_op_type(op_attrs,
                                             OperatorType::REPLICATE)) {
      return DynamicNodeInvocation{
          /*inputs=*/{
              transform(invocation.outputs, to_grad),
          },
          /*node_attrs=*/
          pass_expand_node(invocation.node_attrs, DynamicTaskType::BWD),
          /*outputs=*/
          {
              transform(invocation.inputs, to_grad_output),
          },
      };
    } else {
      return DynamicNodeInvocation{
          /*inputs=*/
          merge_disjoint_maps(std::vector{
              transform(invocation.inputs, to_fwd),
              transform(invocation.outputs, to_fwd),
              transform(invocation.outputs, to_grad),
          }),
          /*node_attrs=*/
          pass_expand_node(invocation.node_attrs, DynamicTaskType::BWD),
          /*outputs=*/
          transform(invocation.inputs, to_grad_output),
      };
    };
  }();

  require_invocation_is_fully_pass_expanded(result);

  return result;
}

std::optional<DynamicNodeMapping> mapping_for_gradient_reduction(
    DynamicOpenDataflowGraph const &g,
    DynamicValueAttrs const &value,
    std::map<dynamic_invocation_id_t, subgradient_id_t> const
        &subgradient_inputs) {
  DynamicSlotSite source_site = dynamic_graph_find_source_of_value(g, value);
  std::set<InternalDynamicSlotSite> sink_sites =
      dynamic_graph_find_sinks_of_value(g, value);

  DynamicNodeInvocation source = dynamic_graph_get_invocation_for_id(
      g, source_site.require_internal().invocation_id);

  if (!source.node_attrs.mapping.has_value()) {
    return std::nullopt;
  }
  DynamicNodeMapping source_mapping = assert_unwrap(source.node_attrs.mapping);

  DynamicTensorSlot source_slot =
      get_only(invert_map(source.outputs).at(value));

  std::pair<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>
      source_tensor_binding = get_only(get_tensor_bindings_for_slot_name(
          source_mapping.op_task_group, source_slot.slot_name));

  // For now we're going to map everything like the original source, both inputs
  // and outputs. This means that in the presence of model parallelism we'll be
  // potentially inducing extra copies that would otherwise be unnecessary.
  std::set<subgradient_id_t> subgradient_ids =
      set_of(values(subgradient_inputs));
  std::map<TensorSlotName, ParallelTensorSpaceCoordinate> input_bindings =
      map_from_pairs(transform(
          subgradient_ids, [&](subgradient_id_t const &subgradient_id) {
            return std::pair{subgradient_id.gradient_reduction_slot,
                             source_tensor_binding.first};
          }));
  std::map<TensorSlotName, ParallelTensorSpaceCoordinate> output_bindings{
      {TensorSlotName::OUTPUT, source_tensor_binding.first},
  };
  std::map<TensorSlotName, ParallelTensorSpaceCoordinate> bindings =
      binary_merge_disjoint_maps(input_bindings, output_bindings);

  DynamicNodeMapping result{
      /*op_task_group=*/MappedOperatorTaskGroup{
          bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding>{
              {source_tensor_binding.second,
               OperatorAtomicTaskShardBinding{bindings}},
          }},
      /*device_type=*/source_mapping.device_type};
  return result;
}

DynamicNodeInvocation create_gradient_reduction_for_value(
    DynamicOpenDataflowGraph const &g,
    DynamicValueAttrs const &value,
    std::map<dynamic_invocation_id_t, subgradient_id_t> const
        &subgradient_inputs) {
  DynamicSlotSite source_site = dynamic_graph_find_source_of_value(g, value);
  std::set<InternalDynamicSlotSite> sink_sites =
      dynamic_graph_find_sinks_of_value(g, value);

  auto to_grad = [](DynamicTensorSlot const &k, DynamicValueAttrs const &v) {
    return std::pair{
        pass_expand_slot(k, FwbTensorType::GRADIENT),
        pass_expand_value(v, FwbTensorType::GRADIENT),
    };
  };

  auto to_grad_subgradient = [](DynamicTensorSlot const &k,
                                DynamicValueAttrs const &v,
                                subgradient_id_t const &s) {
    return std::pair{
        pass_expand_slot(k, FwbTensorType::GRADIENT),
        decide_dynamic_value_attrs_subgradient_id(
            pass_expand_value(v, FwbTensorType::GRADIENT), s),
    };
  };

  DynamicNodeAttrs gradient_reduction_attrs{
      /*task_type=*/DynamicTaskType::BWD,
      /*device_ids=*/std::nullopt,
      /*mapping=*/mapping_for_gradient_reduction(g, value, subgradient_inputs),
      /*op_attrs=*/
      TrainingOperationAttrs{GradientReductionAttrs{}},
      /*layer_guid=*/
      dynamic_layer_guid_t{dynamic_gradient_reduction_layer_guid_t{}},
      /*per_device_op_state=*/std::nullopt,
  };

  DynamicTensorSlot output_slot{
      /*slot_name=*/TensorSlotName::OUTPUT,
      /*slot_tensor_role=*/std::nullopt,
      /*task_shard=*/std::nullopt,
  };

  return DynamicNodeInvocation{
      /*inputs=*/map_from_pairs(transform(
          values(subgradient_inputs),
          [&](subgradient_id_t const &subgradient_id) {
            DynamicTensorSlot input_slot{
                /*slot_name=*/subgradient_id.gradient_reduction_slot,
                /*slot_tensor_role=*/std::nullopt,
                /*task_shard=*/std::nullopt,
            };
            return to_grad_subgradient(input_slot, value, subgradient_id);
          })),
      /*node_attrs=*/gradient_reduction_attrs,
      /*outputs=*/
      {
          to_grad(output_slot, value),
      },
  };
}

DynamicOpenDataflowGraph
    perform_pass_expansion(DynamicOpenDataflowGraph const &g) {

  require_graph_is_ready_for_pass_expansion(g);

  std::set<dynamic_invocation_id_t> needed_in_bwd_pass =
      determine_invocations_needed_in_backward_pass_for_gradient_computation(g);

  std::set<DynamicValueAttrs> values_requiring_gradient_reduction =
      determine_values_requiring_gradient_reduction(g, needed_in_bwd_pass);

  std::map<DynamicValueAttrs,
           std::map<dynamic_invocation_id_t, subgradient_id_t>>
      subgradient_ids_by_value =
          compute_subgradient_ids(g, values_requiring_gradient_reduction);

  DynamicOpenDataflowGraph result = flatmap_dynamic_invocation_set(
      g, [&](DynamicNodeInvocation const &invocation) {
        dynamic_invocation_id_t invocation_id =
            dynamic_graph_get_id_for_invocation(g, invocation);

        std::set<DynamicNodeInvocation> gradient_reductions =
            transform(set_intersection(keys(subgradient_ids_by_value),
                                       set_of(values(invocation.outputs))),
                      [&](DynamicValueAttrs const &v) {
                        return create_gradient_reduction_for_value(
                            g, v, subgradient_ids_by_value.at(v));
                      });

        if (contains(needed_in_bwd_pass, invocation_id)) {
          return set_union(
              std::set{
                  perform_fwd_pass_expansion_for_invocation(invocation),
                  perform_bwd_pass_expansion_for_invocation(
                      invocation_id, invocation, subgradient_ids_by_value),
              },
              gradient_reductions);
        } else {
          return set_union(
              std::set{
                  perform_fwd_pass_expansion_for_invocation(invocation),
              },
              gradient_reductions);
        }
      });

  require_graph_is_fully_pass_expanded(result);

  return result;
}

} // namespace FlexFlow
