#include "task-spec/dynamic_graph/copy_insertion.h"
#include "op-attrs/parallel_tensor_space_coordinate.dtg.h"
#include "op-attrs/tensor_slot_name.dtg.h"
#include "pcg/machine_space_coordinate.dtg.h"
#include "pcg/mapped_parallel_computation_graph/mapped_operator_task_group.h"
#include "task-spec/dynamic_graph/copy_insertion.h"
#include "task-spec/dynamic_graph/dynamic_node_attrs.dtg.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.h"
#include "task-spec/dynamic_graph/dynamic_node_mapping.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_slot_site.dtg.h"
#include "task-spec/dynamic_graph/dynamic_task_type.h"
#include "task-spec/dynamic_graph/dynamic_tensor_slot.dtg.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.h"
#include "task-spec/dynamic_graph/parallel_tensor_mapping.dtg.h"
#include "task-spec/dynamic_graph/training_operation_attrs.h"
#include "utils/bidict/algorithms/bidict_from_unstructured_relation.h"
#include "utils/bidict/algorithms/unstructured_relation_from_bidict.h"
#include "utils/containers/binary_merge_disjoint_maps.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/count.h"
#include "utils/containers/filter_values.h"
#include "utils/containers/filtermap_keys.h"
#include "utils/containers/filtrans.h"
#include "utils/containers/flatmap.h"
#include "utils/containers/get_only.h"
#include "utils/containers/map_values2.h"
#include "utils/containers/merge_disjoint_maps.h"
#include "utils/containers/set_difference.h"
#include "utils/containers/set_intersection.h"
#include "utils/containers/transform.h"
#include "utils/containers/values.h"
#include "utils/containers/zip_values_strict_with.h"
#include "utils/optional.h"
#include "utils/overload.h"

namespace FlexFlow {

bool node_is_copy(DynamicNodeAttrs const &n) {
  return n.op_attrs.has_value() && n.op_attrs.value().is_copy();
}

bool value_is_mapped(DynamicValueAttrs const &n) {
  return n.mapping.has_value();
}

void require_node_is_ready_for_copy_insertion(
    DynamicNodeAttrs const &node_attrs) {
  ASSERT(node_attrs.op_attrs.has_value());
  ASSERT(node_attrs.mapping.has_value());
}

void require_value_is_ready_for_copy_insertion(DynamicValueAttrs const &v) {
  ASSERT(!v.mapping.has_value(), v);
}

void require_invocation_is_ready_for_copy_insertion(
    DynamicNodeInvocation const &i) {
  auto require_slot_is_ready_for_copy_insertion =
      [](DynamicTensorSlot const &) { return; };

  require_invocation_fully_satisfies(i,
                                     require_node_is_ready_for_copy_insertion,
                                     require_value_is_ready_for_copy_insertion,
                                     require_slot_is_ready_for_copy_insertion);
}

void require_graph_is_ready_for_copy_insertion(
    DynamicOpenDataflowGraph const &g) {
  require_full_dynamic_graph_satisfies(
      g, require_invocation_is_ready_for_copy_insertion);
}

void require_value_is_copy_inserted(DynamicValueAttrs const &v) {
  ASSERT(v.mapping.has_value());
}

void require_invocation_is_fully_copy_inserted(DynamicNodeInvocation const &i) {
  auto require_node_is_copy_inserted = [](DynamicNodeAttrs const &) { return; };

  auto require_slot_is_copy_inserted = [](DynamicTensorSlot const &) {
    return;
  };

  require_invocation_fully_satisfies(i,
                                     require_node_is_copy_inserted,
                                     require_value_is_copy_inserted,
                                     require_slot_is_copy_inserted);
}

void require_graph_is_fully_copy_inserted(DynamicOpenDataflowGraph const &g) {
  require_full_dynamic_graph_satisfies(
      g, require_invocation_is_fully_copy_inserted);
}

std::map<DynamicTensorSlot, ParallelTensorMapping>
    get_mappings_for_invocation_id(
        dynamic_invocation_id_t const &i,
        std::map<InternalDynamicSlotSite, ParallelTensorMapping> const
            &mappings) {
  return filtermap_keys(mappings,
                        [&](InternalDynamicSlotSite const &s)
                            -> std::optional<DynamicTensorSlot> {
                          if (s.invocation_id == i) {
                            return s.slot_name;
                          } else {
                            return std::nullopt;
                          }
                        });
}

DynamicNodeInvocation apply_mappings_for_invocation(
    dynamic_invocation_id_t const &id,
    DynamicNodeInvocation const &i,
    std::map<InternalDynamicSlotSite, ParallelTensorMapping> const
        &all_mappings) {

  require_invocation_is_ready_for_copy_insertion(i);

  std::map<DynamicTensorSlot, ParallelTensorMapping> i_mappings =
      get_mappings_for_invocation_id(id, all_mappings);

  std::map<DynamicTensorSlot, ParallelTensorMapping> i_input_mappings =
      restrict_keys(i_mappings, keys(i.inputs));

  std::map<DynamicTensorSlot, ParallelTensorMapping> i_output_mappings =
      restrict_keys(i_mappings, keys(i.outputs));

  auto apply_mapping =
      [&](DynamicValueAttrs const &v,
          ParallelTensorMapping const &mapping) -> DynamicValueAttrs {
    return decide_dynamic_value_attrs_mapping(v, mapping);
  };

  DynamicNodeInvocation result = DynamicNodeInvocation{
      /*inputs=*/
      zip_values_strict_with(i.inputs, i_input_mappings, apply_mapping),
      /*node_attrs=*/
      i.node_attrs,
      /*outputs=*/
      zip_values_strict_with(i.outputs, i_output_mappings, apply_mapping),
  };

  require_invocation_is_fully_copy_inserted(result);

  return result;
}

DynamicNodeInvocation
    make_copy_invocation(DynamicValueCopyInfo const &copy_info) {
  DynamicNodeInvocation result = DynamicNodeInvocation{
      /*inputs=*/{
          {
              DynamicTensorSlot{
                  /*slot_name=*/TensorSlotName::INPUT,
                  /*slot_tensor_role=*/std::nullopt,
                  /*task_shard=*/std::nullopt,
              },
              decide_dynamic_value_attrs_mapping(copy_info.value_attrs,
                                                 copy_info.src_mapping),
          },
      },
      /*node_attrs=*/
      DynamicNodeAttrs{
          /*task_type=*/std::nullopt,
          /*device_coord=*/std::nullopt,
          /*mapping=*/std::nullopt,
          /*op_attrs*/ TrainingOperationAttrs{CopyAttrs{}},
          /*layer_guid=*/dynamic_layer_guid_t{dynamic_copy_layer_guid_t{}},
          /*per_device_op_state=*/std::nullopt,
      },
      /*outputs=*/
      {
          {
              DynamicTensorSlot{
                  /*slot_name=*/TensorSlotName::OUTPUT,
                  /*slot_tensor_role=*/std::nullopt,
                  /*task_shard=*/std::nullopt,
              },
              decide_dynamic_value_attrs_mapping(copy_info.value_attrs,
                                                 copy_info.dst_mapping),
          },
      },
  };

  require_invocation_is_fully_copy_inserted(result);

  return result;
}

std::set<DynamicValueCopyInfo> copies_for_value(
    DynamicValueAttrs const &v,
    DynamicSlotSite const &src_site,
    std::set<InternalDynamicSlotSite> const &dst_sites,
    std::map<InternalDynamicSlotSite, ParallelTensorMapping> const
        &all_mappings) {

  require_value_is_ready_for_copy_insertion(v);

  return src_site.visit<std::set<DynamicValueCopyInfo>>(overload{
      [&](ExternalDynamicSlotSite const &) -> std::set<DynamicValueCopyInfo> {
        return {};
      },
      [&](InternalDynamicSlotSite const &s) -> std::set<DynamicValueCopyInfo> {
        ParallelTensorMapping src_mapping = all_mappings.at(s);
        std::map<InternalDynamicSlotSite, ParallelTensorMapping>
            sink_site_mappings = restrict_keys(all_mappings, dst_sites);

        return copies_for_internal_value(v, s, src_mapping, sink_site_mappings);
      }});
}

std::set<DynamicValueCopyInfo> copies_for_internal_value(
    DynamicValueAttrs const &v,
    InternalDynamicSlotSite const &src_site,
    ParallelTensorMapping const &src_mapping,
    std::map<InternalDynamicSlotSite, ParallelTensorMapping> const
        &sink_site_mappings) {

  require_value_is_ready_for_copy_insertion(v);

  std::set<ParallelTensorMapping> sink_mapping_set =
      set_of(values(sink_site_mappings));

  std::set<ParallelTensorMapping> required_copies =
      set_difference(sink_mapping_set, std::set{src_mapping});

  auto make_copy_to =
      [&](ParallelTensorMapping const &sink_mapping) -> DynamicValueCopyInfo {
    return DynamicValueCopyInfo{
        /*value_attrs=*/v,
        /*src_mapping=*/src_mapping,
        /*sink_mapping=*/sink_mapping,
    };
  };

  return transform(required_copies, make_copy_to);
}

std::map<InternalDynamicSlotSite, ParallelTensorMapping>
    resolve_tensor_mappings(DynamicOpenDataflowGraph const &g) {
  require_graph_is_ready_for_copy_insertion(g);

  std::map<InternalDynamicSlotSite, ParallelTensorMapping>
      resolved_from_node_mappings =
          resolve_partial_tensor_mappings_from_node_mappings(g);

  std::map<InternalDynamicSlotSite, ParallelTensorMapping>
      resolved_from_adjacent_values =
          resolve_missing_tensor_mappings_from_adjacent_values(
              g, resolved_from_node_mappings);

  std::map<InternalDynamicSlotSite, ParallelTensorMapping> result =
      binary_merge_disjoint_maps(resolved_from_node_mappings,
                                 resolved_from_adjacent_values);

  {
    std::set<InternalDynamicSlotSite> all_internal_slot_sites =
        get_internal_dynamic_slot_sites(g);
    std::set<InternalDynamicSlotSite> resolved_slot_sites = keys(result);
    ASSERT(resolved_slot_sites == all_internal_slot_sites);
  }

  return result;
}

std::map<InternalDynamicSlotSite, ParallelTensorMapping>
    resolve_partial_tensor_mappings_from_node_mappings(
        DynamicOpenDataflowGraph const &g) {
  require_graph_is_ready_for_copy_insertion(g);

  auto slots_to_map_for_replicate =
      [](dynamic_invocation_id_t const &invocation_id,
         DynamicNodeInvocation const &invocation)
      -> std::set<InternalDynamicSlotSite> {
    TrainingOpType op_type = dynamic_node_invocation_get_op_type(invocation);

    ASSERT(op_type == TrainingOpType{OperatorType::REPLICATE});

    std::set<InternalDynamicSlotSite> slot_sites =
        (invocation.node_attrs.task_type == DynamicTaskType::BWD)
            ? get_incoming_dynamic_slot_sites_for_invocation(invocation_id,
                                                             invocation)
            : get_output_dynamic_slot_sites_for_invocation(invocation_id,
                                                           invocation);

    {
      InternalDynamicSlotSite slot_site = get_only(slot_sites);
      ASSERT(slot_site.slot_name.slot_name == TensorSlotName::OUTPUT);
    }

    return slot_sites;
  };

  auto get_mappings_for_invocation =
      [&](DynamicNodeInvocation const &invocation)
      -> std::map<InternalDynamicSlotSite, ParallelTensorMapping> {
    TrainingOpType op_type = dynamic_node_invocation_get_op_type(invocation);
    dynamic_invocation_id_t invocation_id =
        dynamic_graph_get_id_for_invocation(g, invocation);

    std::set<InternalDynamicSlotSite> slot_sites_to_resolve = [&] {
      if (op_type == TrainingOpType{OperatorType::REPLICATE}) {
        return slots_to_map_for_replicate(invocation_id, invocation);
      } else {
        return get_dynamic_slot_sites_for_invocation(invocation_id, invocation);
      }
    }();

    return generate_map(
        slot_sites_to_resolve,
        [&](InternalDynamicSlotSite const &s) -> ParallelTensorMapping {
          return ParallelTensorMapping{
              dynamic_node_mapping_bindings_for_slot_name(
                  assert_unwrap(invocation.node_attrs.mapping),
                  s.slot_name.slot_name),
          };
        });
  };

  std::map<InternalDynamicSlotSite, ParallelTensorMapping> result =
      merge_disjoint_maps(transform(get_dynamic_invocation_set(g),
                                    get_mappings_for_invocation));

  return result;
}

std::map<InternalDynamicSlotSite, ParallelTensorMapping>
    resolve_missing_tensor_mappings_from_adjacent_values(
        DynamicOpenDataflowGraph const &g,
        std::map<InternalDynamicSlotSite, ParallelTensorMapping> const
            &resolved_mappings) {
  require_graph_is_ready_for_copy_insertion(g);

  std::set<InternalDynamicSlotSite> all_internal_slot_sites =
      get_internal_dynamic_slot_sites(g);

  std::set<InternalDynamicSlotSite> missing_mappings =
      set_minus(all_internal_slot_sites, keys(resolved_mappings));

  auto get_mapping_for_slot_site_from_adjacent_values =
      [&](InternalDynamicSlotSite const &slot_site) -> ParallelTensorMapping {
    DynamicNodeInvocation invocation =
        dynamic_graph_get_invocation_for_id(g, slot_site.invocation_id);

    TrainingOpType op_type = dynamic_node_invocation_get_op_type(invocation);
    std::optional<DynamicTaskType> task_type = invocation.node_attrs.task_type;

    TrainingOpType replicate_op_type = TrainingOpType{OperatorType::REPLICATE};

    if (op_type == replicate_op_type && task_type == DynamicTaskType::BWD) {
      ASSERT(slot_site.direction == TensorDirection::OUTPUT);

      InternalDynamicSlotSite slot_site_sink =
          get_only(dynamic_graph_find_sinks_of_slot_site(g, slot_site));

      ASSERT(contains_key(resolved_mappings, slot_site_sink));

      return resolved_mappings.at(slot_site_sink);
    } else if (op_type == replicate_op_type &&
               (task_type == std::nullopt ||
                task_type == DynamicTaskType::FWD)) {
      ASSERT(slot_site.direction == TensorDirection::INCOMING);

      InternalDynamicSlotSite slot_site_src =
          dynamic_graph_find_source_of_slot_site(g, slot_site)
              .require_internal();

      ASSERT(contains_key(resolved_mappings, slot_site_src));

      return resolved_mappings.at(slot_site_src);
    } else {
      PANIC("Unhandled case");
    }
  };

  return generate_map(missing_mappings,
                      get_mapping_for_slot_site_from_adjacent_values);
}

std::set<DynamicValueCopyInfo>
    infer_all_copies_in_graph(DynamicOpenDataflowGraph const &g) {
  std::map<InternalDynamicSlotSite, ParallelTensorMapping>
      fully_resolved_tensor_mappings = resolve_tensor_mappings(g);

  std::set<DynamicValueCopyInfo> all_copies = flatmap(
      set_of(get_dynamic_values(g)),
      [&](DynamicValueAttrs const &v) -> std::set<DynamicValueCopyInfo> {
        DynamicSlotSite src_site = dynamic_graph_find_source_of_value(g, v);
        std::set<InternalDynamicSlotSite> sinks =
            dynamic_graph_find_sinks_of_value(g, v);

        return copies_for_value(
            v, src_site, sinks, fully_resolved_tensor_mappings);
      });

  return all_copies;
}

DynamicOpenDataflowGraph
    perform_copy_insertion(DynamicOpenDataflowGraph const &g) {

  require_graph_is_ready_for_copy_insertion(g);

  std::map<InternalDynamicSlotSite, ParallelTensorMapping>
      fully_resolved_tensor_mappings = resolve_tensor_mappings(g);

  std::set<DynamicValueCopyInfo> all_copies = infer_all_copies_in_graph(g);

  std::set<DynamicNodeInvocation> all_copy_invocations =
      transform(all_copies, make_copy_invocation);

  std::set<DynamicNodeInvocation> mapped_invocations =
      transform(get_dynamic_invocation_set(g),
                [&](DynamicNodeInvocation const &i) -> DynamicNodeInvocation {
                  dynamic_invocation_id_t id =
                      dynamic_graph_get_id_for_invocation(g, i);

                  return apply_mappings_for_invocation(
                      id, i, fully_resolved_tensor_mappings);
                });

  DynamicOpenDataflowGraph result =
      dynamic_open_dataflow_graph_from_invocation_set(
          set_union(all_copy_invocations, mapped_invocations));

  require_graph_is_fully_copy_inserted(result);

  return result;
}

} // namespace FlexFlow
