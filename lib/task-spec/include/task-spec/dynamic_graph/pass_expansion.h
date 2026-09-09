#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_PASS_EXPANSION_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_PASS_EXPANSION_H

#include "task-spec/dynamic_graph/dynamic_invocation_id_t.dtg.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.dtg.h"

namespace FlexFlow {

void require_node_might_be_pass_expanded(DynamicNodeAttrs const &);
void require_node_might_not_be_pass_expanded(DynamicNodeAttrs const &);

void require_slot_is_not_pass_expanded(DynamicTensorSlot const &);

void require_value_is_pass_expanded(DynamicValueAttrs const &);
void require_value_is_not_pass_expanded(DynamicValueAttrs const &);

void require_invocation_is_fully_pass_expanded(DynamicNodeInvocation const &);
void require_invocation_is_ready_for_pass_expansion(
    DynamicNodeInvocation const &);

void require_graph_is_fully_pass_expanded(DynamicOpenDataflowGraph const &);
void require_graph_is_ready_for_pass_expansion(
    DynamicOpenDataflowGraph const &);

std::set<DynamicValueAttrs>
    determine_intermediate_values_needed_to_compute_gradients_of_value(
        DynamicOpenDataflowGraph const &, DynamicValueAttrs const &);

std::set<DynamicValueAttrs>
    determine_intermediate_values_needed_for_gradient_computation(
        DynamicOpenDataflowGraph const &);

std::set<dynamic_invocation_id_t>
    determine_invocations_needed_in_backward_pass_for_gradient_computation(
        DynamicOpenDataflowGraph const &);

DynamicTensorSlot pass_expand_slot(DynamicTensorSlot const &, FwbTensorType);
DynamicValueAttrs pass_expand_value(DynamicValueAttrs const &, FwbTensorType);
DynamicNodeAttrs pass_expand_node(DynamicNodeAttrs const &, DynamicTaskType);

DynamicNodeInvocation
    perform_fwd_pass_expansion_for_invocation(DynamicNodeInvocation const &);
DynamicNodeInvocation perform_bwd_pass_expansion_for_invocation(
    dynamic_invocation_id_t const &,
    DynamicNodeInvocation const &,
    std::map<DynamicValueAttrs,
             std::map<dynamic_invocation_id_t, subgradient_id_t>> const &);
DynamicNodeInvocation create_gradient_reduction_for_value(
    DynamicOpenDataflowGraph const &,
    DynamicValueAttrs const &,
    std::map<dynamic_invocation_id_t, subgradient_id_t> const &);

DynamicOpenDataflowGraph
    perform_pass_expansion(DynamicOpenDataflowGraph const &);

} // namespace FlexFlow

#endif
