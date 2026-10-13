#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_OPEN_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_OPEN_DATAFLOW_GRAPH_H

#include "task-spec/dynamic_graph/dynamic_graph_edge.dtg.h"
#include "task-spec/dynamic_graph/dynamic_invocation_id_t.dtg.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.dtg.h"
#include "task-spec/dynamic_graph/dynamic_slot_site.dtg.h"
#include "task-spec/dynamic_graph/dynamic_value_id_t.dtg.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/labelled_open_kwarg_dataflow_graph.h"

namespace FlexFlow {

DynamicOpenDataflowGraph make_empty_dynamic_open_dataflow_graph();

void check_dynamic_open_dataflow_graph_is_valid(
    DynamicOpenDataflowGraph const &);

DynamicOpenDataflowGraph compute_invocation_ids_for_dynamic_open_dataflow_graph(
    DynamicOpenDataflowGraph const &);

DynamicOpenDataflowGraph compute_value_ids_for_dynamic_open_dataflow_graph(
    DynamicOpenDataflowGraph const &);

nonnegative_int dynamic_graph_num_nodes(DynamicOpenDataflowGraph const &);

bool full_dynamic_graph_satisfies(
    DynamicOpenDataflowGraph const &,
    std::function<bool(DynamicNodeAttrs const &)> const &,
    std::function<bool(DynamicValueAttrs const &)> const &,
    std::function<bool(DynamicTensorSlot const &)> const &);

bool no_part_of_dynamic_graph_satisfies(
    DynamicOpenDataflowGraph const &,
    std::function<bool(DynamicNodeAttrs const &)> const &,
    std::function<bool(DynamicValueAttrs const &)> const &,
    std::function<bool(DynamicTensorSlot const &)> const &);

void require_full_dynamic_graph_satisfies(
    DynamicOpenDataflowGraph const &,
    std::function<void(DynamicNodeInvocation const &)> const &);

std::multiset<DynamicNodeAttrs>
    get_dynamic_nodes(DynamicOpenDataflowGraph const &);
std::multiset<DynamicValueAttrs>
    get_dynamic_values(DynamicOpenDataflowGraph const &);
std::multiset<DynamicTensorSlot>
    get_dynamic_tensor_slots(DynamicOpenDataflowGraph const &);
std::set<DynamicNodeInvocation>
    get_dynamic_invocation_set(DynamicOpenDataflowGraph const &);

std::set<DynamicValueAttrs>
    dynamic_graph_get_internal_values(DynamicOpenDataflowGraph const &);
std::set<DynamicValueAttrs>
    dynamic_graph_get_external_values(DynamicOpenDataflowGraph const &);

dynamic_invocation_id_t
    dynamic_graph_get_id_for_invocation(DynamicOpenDataflowGraph const &,
                                        DynamicNodeInvocation const &);
DynamicNodeInvocation
    dynamic_graph_get_invocation_for_id(DynamicOpenDataflowGraph const &,
                                        dynamic_invocation_id_t const &);

dynamic_value_id_t
    dynamic_graph_get_id_for_value(DynamicOpenDataflowGraph const &,
                                   DynamicValueAttrs const &);
DynamicValueAttrs
    dynamic_graph_get_value_for_id(DynamicOpenDataflowGraph const &,
                                   dynamic_value_id_t const &);

std::set<DynamicGraphEdge>
    get_dynamic_graph_edges(DynamicOpenDataflowGraph const &);
std::set<DynamicGraphEdge> get_dynamic_graph_edges_incoming_to_invocation(
    DynamicOpenDataflowGraph const &, DynamicNodeInvocation const &);
std::set<DynamicGraphEdge> get_dynamic_graph_edges_outgoing_from_invocation(
    DynamicOpenDataflowGraph const &, DynamicNodeInvocation const &);

std::set<InternalDynamicSlotSite>
    get_internal_dynamic_slot_sites(DynamicOpenDataflowGraph const &);

std::set<DynamicSlotSite>
    get_dynamic_slot_sites(DynamicOpenDataflowGraph const &);

DynamicSlotSite
    dynamic_graph_find_source_of_slot_site(DynamicOpenDataflowGraph const &,
                                           InternalDynamicSlotSite const &);
std::set<InternalDynamicSlotSite>
    dynamic_graph_find_sinks_of_slot_site(DynamicOpenDataflowGraph const &,
                                          InternalDynamicSlotSite const &);

DynamicSlotSite
    dynamic_graph_find_source_of_value(DynamicOpenDataflowGraph const &,
                                       DynamicValueAttrs const &);
std::set<InternalDynamicSlotSite>
    dynamic_graph_find_sinks_of_value(DynamicOpenDataflowGraph const &,
                                      DynamicValueAttrs const &);

DynamicValueAttrs
    dynamic_value_attrs_for_slot_site(DynamicOpenDataflowGraph const &,
                                      DynamicSlotSite const &);

std::optional<DynamicValueAttrs>
    find_output_value_attrs(DynamicOpenDataflowGraph const &,
                            dynamic_tensor_guid_t,
                            std::optional<DynamicTensorRole> const &);

DynamicOpenDataflowGraph transform_dynamic_invocation_set(
    DynamicOpenDataflowGraph const &,
    std::function<DynamicNodeInvocation(DynamicNodeInvocation const &)> const
        &);

DynamicOpenDataflowGraph flatmap_dynamic_invocation_set(
    DynamicOpenDataflowGraph const &,
    std::function<std::set<DynamicNodeInvocation>(
        DynamicNodeInvocation const &)> const &);

DynamicOpenDataflowGraph dynamic_open_dataflow_graph_from_invocation_set(
    std::set<DynamicNodeInvocation> const &);

std::pair<LabelledOpenKwargDataflowGraph<DynamicNodeAttrs,
                                         DynamicValueAttrs,
                                         int,
                                         DynamicTensorSlot>,
          bidict<Node, DynamicNodeInvocation>>
    labelled_open_kwarg_dataflow_graph_from_dynamic_open_dataflow_graph(
        DynamicOpenDataflowGraph const &);

bool dynamic_open_dataflow_graphs_are_isomorphic(
    DynamicOpenDataflowGraph const &, DynamicOpenDataflowGraph const &);

std::string
    dynamic_open_dataflow_graph_as_dot(DynamicOpenDataflowGraph const &);
void debug_print_dynamic_open_dataflow_graph_as_dot(
    DynamicOpenDataflowGraph const &);

} // namespace FlexFlow

#endif
