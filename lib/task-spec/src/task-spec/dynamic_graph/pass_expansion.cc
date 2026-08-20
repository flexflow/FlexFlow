#include "task-spec/dynamic_graph/pass_expansion.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_tensor_role.h"
#include "task-spec/dynamic_graph/parallel_op_data_movement.h"
#include "task-spec/dynamic_graph/training_operation_attrs.h"
#include "utils/containers/are_all_same.h"
#include "utils/containers/flatmap.h"
#include "utils/containers/get_only.h"
#include "utils/containers/map_values.h"
#include "utils/containers/merge_disjoint_maps.h"
#include "utils/containers/repeat_until_converged.h"
#include "utils/containers/transform.h"

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
}

void require_slot_is_not_pass_expanded(DynamicTensorSlot const &s) {
  ASSERT(!s.slot_tensor_role.has_value(), s);
}

void require_value_is_pass_expanded(DynamicValueAttrs const &v) {
  ASSERT(v.role.has_value(), v);
}

void require_value_is_not_pass_expanded(DynamicValueAttrs const &v) {
  ASSERT(!v.role.has_value(), v);
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
    DynamicNodeInvocation const &invocation) {

  require_invocation_is_ready_for_pass_expansion(invocation);

  TrainingOperationAttrs op_attrs =
      assert_unwrap(invocation.node_attrs.op_attrs);

  auto to_fwd = [](DynamicTensorSlot const &k, DynamicValueAttrs const &v) {
    return std::pair{
        pass_expand_slot(k, FwbTensorType::FORWARD),
        pass_expand_value(v, FwbTensorType::FORWARD),
    };
  };

  auto to_grad_value = [](DynamicValueAttrs const &v) {
    return pass_expand_value(v, FwbTensorType::GRADIENT);
  };

  auto to_grad = [](DynamicTensorSlot const &k, DynamicValueAttrs const &v) {
    return std::pair{
        pass_expand_slot(k, FwbTensorType::GRADIENT),
        pass_expand_value(v, FwbTensorType::GRADIENT),
    };
  };

  DynamicNodeInvocation result = [&]() -> DynamicNodeInvocation {
    if (op_attrs.is_copy()) {
      return DynamicNodeInvocation{
          /*inputs=*/map_values(invocation.outputs, to_grad_value),
          /*node_attrs=*/invocation.node_attrs,
          /*outputs=*/map_values(invocation.inputs, to_grad_value),
      };
    } else if (is_parallel_training_op(op_attrs)) {
      // All parallel ops: BWD is purely gradient data movement.
      // No fwd activations are needed — just swap input/output grad roles.
      return DynamicNodeInvocation{
          /*inputs=*/{
              transform(invocation.outputs, to_grad),
          },
          /*node_attrs=*/
          pass_expand_node(invocation.node_attrs, DynamicTaskType::BWD),
          /*outputs=*/
          {
              transform(invocation.inputs, to_grad),
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
          transform(invocation.inputs, to_grad),
      };
    };
  }();

  require_invocation_is_fully_pass_expanded(result);

  return result;
}

DynamicOpenDataflowGraph
    perform_pass_expansion(DynamicOpenDataflowGraph const &g) {

  require_graph_is_ready_for_pass_expansion(g);

  std::set<dynamic_invocation_id_t> needed_in_bwd_pass =
      determine_invocations_needed_in_backward_pass_for_gradient_computation(g);

  DynamicOpenDataflowGraph result = flatmap_dynamic_invocation_set(
      g, [&](DynamicNodeInvocation const &invocation) {
        dynamic_invocation_id_t invocation_id =
            dynamic_graph_get_id_for_invocation(g, invocation);

        if (contains(needed_in_bwd_pass, invocation_id)) {
          return std::set{
              perform_fwd_pass_expansion_for_invocation(invocation),
              perform_bwd_pass_expansion_for_invocation(invocation),
          };
        } else {
          return std::set{
              perform_fwd_pass_expansion_for_invocation(invocation),
          };
        }
      });

  require_graph_is_fully_pass_expanded(result);

  return result;
}

} // namespace FlexFlow
