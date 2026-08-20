#include "task-spec/dynamic_graph/shard_expansion.h"
#include "pcg/mapped_parallel_computation_graph/operator_atomic_task_shard_binding.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.h"
#include "task-spec/dynamic_graph/dynamic_node_mapping.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_tensor_role.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "task-spec/dynamic_graph/parallel_op_data_movement.h"
#include "task-spec/dynamic_graph/parallel_tensor_mapping.h"
#include "task-spec/dynamic_graph/serializable_dynamic_node_invocation.h"
#include "task-spec/dynamic_graph/shard_expansion.h"
#include "task-spec/dynamic_graph/training_operation_attrs.h"
#include "utils/bidict/algorithms/bidict_filter_values.h"
#include "utils/binary_relation/binary_relation_from_map.h"
#include "utils/binary_relation/binary_relation_transform_right2.h"
#include "utils/binary_relation/filter_binary_relation.h"
#include "utils/containers/are_disjoint.h"
#include "utils/containers/binary_merge_disjoint_maps.h"
#include "utils/containers/flatmap.h"
#include "utils/containers/get_only.h"
#include "utils/containers/map_from_pairs.h"
#include "utils/containers/map_from_unordered.h"
#include "utils/containers/map_keys.h"
#include "utils/containers/merge_disjoint_maps.h"
#include "utils/containers/require_only_key.h"
#include "utils/containers/require_same.h"
#include "utils/containers/set_of.h"
#include "utils/containers/transform.h"
#include "utils/optional.h"

namespace FlexFlow {

void require_node_is_shard_expanded(DynamicNodeAttrs const &n) {
  ASSERT(n.device_ids.has_value());
}

void require_value_is_shard_expanded(DynamicValueAttrs const &n) {
  ASSERT(n.shard_coord.has_value());
}

void require_invocation_is_fully_shard_expanded(
    DynamicNodeInvocation const &i) {
  auto require_slot_is_shard_expanded = [](DynamicTensorSlot const &) {
    return;
  };

  return require_invocation_fully_satisfies(i,
                                            require_node_is_shard_expanded,
                                            require_value_is_shard_expanded,
                                            require_slot_is_shard_expanded);
}

void require_graph_is_fully_shard_expanded(DynamicOpenDataflowGraph const &g) {
  return require_full_dynamic_graph_satisfies(
      g, require_invocation_is_fully_shard_expanded);
}

void require_node_is_ready_for_shard_expansion(DynamicNodeAttrs const &n) {
  ASSERT(n.op_attrs.has_value());

  if (n.op_attrs.value().is_pcg_op()) {
    ASSERT(n.mapping.has_value());
  }
}

void require_value_is_ready_for_shard_expansion(DynamicValueAttrs const &n) {
  return;
}

void require_invocation_is_ready_for_shard_expansion(
    DynamicNodeInvocation const &i) {
  auto require_slot_is_ready_for_shard_expansion =
      [](DynamicTensorSlot const &) { return; };

  return require_invocation_fully_satisfies(
      i,
      require_node_is_ready_for_shard_expansion,
      require_value_is_ready_for_shard_expansion,
      require_slot_is_ready_for_shard_expansion);
}

void require_graph_is_ready_for_shard_expansion(
    DynamicOpenDataflowGraph const &g) {
  require_full_dynamic_graph_satisfies(
      g, require_invocation_is_ready_for_shard_expansion);
}

static DynamicNodeInvocationShardingInfo invocation_sharding_info_for_binding(
    DynamicNodeInvocation const &i,
    global_device_id_t const &device_id,
    OperatorAtomicTaskShardBinding const &binding) {

  auto shard_expand_value_attrs =
      [&](DynamicTensorSlot const &s,
          DynamicValueAttrs const &v) -> DynamicValueAttrsShardingInfo {
    ParallelTensorSpaceCoordinate parallel_tensor_coord =
        binding.tensor_coords.at(s.slot_name);

    return DynamicValueAttrsShardingInfo{
        /*shard_coord=*/parallel_tensor_coord,
        /*mapping=*/
        pt_mapping_get_device_for_coord(assert_unwrap(v.mapping),
                                        parallel_tensor_coord),
    };
  };

  DynamicNodeInvocationShardingInfo result = DynamicNodeInvocationShardingInfo{
      /*device_coord=*/nonempty_set{device_id},
      /*value_sharding=*/
      binary_relation_transform_right2(
          binary_relation_from_map(
              binary_merge_disjoint_maps(i.inputs, i.outputs)),
          shard_expand_value_attrs),
  };

  {
    std::set<DynamicTensorSlot> invocation_slots =
        set_union(keys(i.inputs), keys(i.outputs));

    std::set<DynamicTensorSlot> sharding_info_slots =
        result.value_sharding.left_values();

    ASSERT(invocation_slots == sharding_info_slots);
  }

  return result;
}

static std::set<DynamicNodeInvocationShardingInfo>
    generate_shard_expansion_for_copy(DynamicNodeInvocation const &i) {
  auto [input_slot, input] = get_only(i.inputs);
  auto [output_slot, output] = get_only(i.outputs);

  ParallelTensorMapping input_mapping = assert_unwrap(input.mapping);
  ParallelTensorMapping output_mapping = assert_unwrap(output.mapping);

  std::set<ParallelTensorSpaceCoordinate> coord_set =
      require_same(pt_mapping_get_coord_set(input_mapping),
                   pt_mapping_get_coord_set(output_mapping));

  return transform(
      coord_set,
      [&](ParallelTensorSpaceCoordinate const &p)
          -> DynamicNodeInvocationShardingInfo {
        // The machine coord for a copy is inherently nebulous because it
        // doesn't strictly run in any single location. Further, Realm has the
        // flexibility to issue a copy operation from anywhere in the machine,
        // including remotely. Here we choose machine_coord based on the input
        // because we expect this to align with the most efficient way to issue
        // copies in Realm, although the current Realm backend uses a
        // centralized controller and thus issues copies all from a single node.
        global_device_id_t device_id =
            pt_mapping_get_device_for_coord(input_mapping, p);

        return invocation_sharding_info_for_binding(
            i,
            device_id,
            OperatorAtomicTaskShardBinding{{
                {input_slot.slot_name, p},
                {output_slot.slot_name, p},
            }});
      });
}

static std::set<DynamicNodeInvocationShardingInfo>
    generate_shard_expansion_using_node_mapping(
        DynamicNodeInvocation const &i,
        DynamicTensorSlot const &one_side_slot,
        DynamicTensorSlot const &n_side_slot,
        ParallelTensorMapping const &one_side_mapping,
        ParallelTensorMapping const &n_side_mapping) {

  DynamicNodeMapping const node_mapping = assert_unwrap(i.node_attrs.mapping);
  bidict<global_device_id_t, OperatorAtomicTaskShardBinding> const
      shard_bindings = dynamic_node_mapping_get_shard_bindings(node_mapping);

  std::set<ParallelTensorSpaceCoordinate> const one_side_coords =
      pt_mapping_get_coord_set(one_side_mapping);

  return transform(
      one_side_coords,
      [&](ParallelTensorSpaceCoordinate const &one_coord)
          -> DynamicNodeInvocationShardingInfo {
        // Find all devices whose shard binding has this one-side coord
        bidict<global_device_id_t, OperatorAtomicTaskShardBinding> const
            for_one_coord = bidict_filter_values(
                shard_bindings,
                [&](OperatorAtomicTaskShardBinding const &b) -> bool {
                  return ptensor_space_coord_for_slot_name(
                             b, one_side_slot.slot_name) == one_coord;
                });

        nonempty_set<global_device_id_t> const group_device_ids =
            nonempty_set(for_one_coord.left_values());

        // Build n-side sharding infos — one per device in the group
        std::map<DynamicTensorSlot, DynamicValueAttrsShardingInfo>
            n_side_sharding_infos = map_from_pairs(transform(
                group_device_ids.unwrap_as_set(),
                [&](global_device_id_t const &device)
                    -> std::pair<DynamicTensorSlot,
                                 DynamicValueAttrsShardingInfo> {
                  ParallelTensorSpaceCoordinate const n_coord =
                      ptensor_space_coord_for_slot_name(
                          shard_bindings.at_l(device), n_side_slot.slot_name);
                  return {
                      DynamicTensorSlot{
                          /*slot_name=*/n_side_slot.slot_name,
                          /*slot_tensor_role=*/n_side_slot.slot_tensor_role,
                          /*task_shard=*/device.coord,
                      },
                      DynamicValueAttrsShardingInfo{
                          /*shard_coord=*/n_coord,
                          /*mapping=*/device,
                      },
                  };
                }));

        // Build one-side sharding info — single entry, task_shard=nullopt
        std::map<DynamicTensorSlot, DynamicValueAttrsShardingInfo> const
            one_side_sharding_infos = {
                {
                    DynamicTensorSlot{
                        /*slot_name=*/one_side_slot.slot_name,
                        /*slot_tensor_role=*/one_side_slot.slot_tensor_role,
                        /*task_shard=*/std::nullopt,
                    },
                    DynamicValueAttrsShardingInfo{
                        /*shard_coord=*/one_coord,
                        /*mapping=*/
                        pt_mapping_get_device_for_coord(one_side_mapping,
                                                        one_coord),
                    },
                },
            };

        return DynamicNodeInvocationShardingInfo{
            /*device_ids=*/group_device_ids,
            /*value_sharding=*/
            binary_relation_from_map(binary_merge_disjoint_maps(
                one_side_sharding_infos, n_side_sharding_infos)),
        };
      });
}

static std::set<DynamicNodeInvocationShardingInfo>
    generate_shard_expansion_for_parallel_op(
        DynamicNodeInvocation const &i, ParallelOpMovementKind const kind) {

  DynamicTensorSlot const input_slot = get_only(keys(i.inputs));
  DynamicTensorSlot const output_slot = get_only(keys(i.outputs));
  DynamicValueAttrs const &input = i.inputs.at(input_slot);
  DynamicValueAttrs const &output = i.outputs.at(output_slot);

  ParallelTensorMapping const input_mapping = assert_unwrap(input.mapping);
  ParallelTensorMapping const output_mapping = assert_unwrap(output.mapping);

  std::set<ParallelTensorSpaceCoordinate> const input_coords =
      pt_mapping_get_coord_set(input_mapping);
  std::set<ParallelTensorSpaceCoordinate> const output_coords =
      pt_mapping_get_coord_set(output_mapping);

  if (kind == ParallelOpMovementKind::BROADCAST) {
    // 1 input → N outputs: one-side is input, n-side is output
    return generate_shard_expansion_using_node_mapping(
        i, input_slot, output_slot, input_mapping, output_mapping);
  }

  if (kind == ParallelOpMovementKind::SUM_REDUCE) {
    // N inputs → 1 output: one-side is output, n-side is input
    return generate_shard_expansion_using_node_mapping(
        i, output_slot, input_slot, output_mapping, input_mapping);
  }

  if (kind == ParallelOpMovementKind::RESHUFFLE) {
    DynamicNodeMapping const node_mapping = assert_unwrap(i.node_attrs.mapping);
    bidict<global_device_id_t, OperatorAtomicTaskShardBinding> const
        shard_bindings = dynamic_node_mapping_get_shard_bindings(node_mapping);

    return transform(
        output_coords,
        [&](ParallelTensorSpaceCoordinate const &out_coord)
            -> DynamicNodeInvocationShardingInfo {
          global_device_id_t const out_device =
              pt_mapping_get_device_for_coord(output_mapping, out_coord);

          // Look up the input coord for this device directly by slot name,
          // rather than scanning for a coord value that happens to appear in
          // input_mapping — coordinate values can collide across the input
          // and output coordinate spaces, which would otherwise risk pairing
          // the wrong device.
          OperatorAtomicTaskShardBinding const &binding =
              shard_bindings.at_l(out_device);
          ParallelTensorSpaceCoordinate const in_coord =
              ptensor_space_coord_for_slot_name(binding, input_slot.slot_name);

          global_device_id_t const in_device =
              pt_mapping_get_device_for_coord(input_mapping, in_coord);

          return DynamicNodeInvocationShardingInfo{
              /*device_ids=*/nonempty_set{out_device},
              /*value_sharding=*/
              binary_relation_from_map(
                  std::map<DynamicTensorSlot, DynamicValueAttrsShardingInfo>{
                      {
                          DynamicTensorSlot{
                              /*slot_name=*/input_slot.slot_name,
                              /*slot_tensor_role=*/input_slot.slot_tensor_role,
                              /*task_shard=*/std::nullopt,
                          },
                          DynamicValueAttrsShardingInfo{
                              /*shard_coord=*/in_coord,
                              /*mapping=*/in_device,
                          },
                      },
                      {
                          DynamicTensorSlot{
                              /*slot_name=*/output_slot.slot_name,
                              /*slot_tensor_role=*/output_slot.slot_tensor_role,
                              /*task_shard=*/std::nullopt,
                          },
                          DynamicValueAttrsShardingInfo{
                              /*shard_coord=*/out_coord,
                              /*mapping=*/out_device,
                          },
                      },
                  }),
          };
        });
  }

  // GATHER: N inputs → 1 output (copy).
  // Anchor on the single output coord; all input coords feed into it.
  // input_coords are unique across shard bindings (resolved from node mapping).
  ASSERT(kind == ParallelOpMovementKind::GATHER);

  auto make_sharding_infos =
      [](std::set<ParallelTensorSpaceCoordinate> const &coords,
         ParallelTensorMapping const &mapping,
         DynamicTensorSlot const &base_slot)
      -> std::map<DynamicTensorSlot, DynamicValueAttrsShardingInfo> {
    bool const needs_task_shard = coords.size() > 1;
    return map_from_pairs(transform(
        coords,
        [&](ParallelTensorSpaceCoordinate const &coord)
            -> std::pair<DynamicTensorSlot, DynamicValueAttrsShardingInfo> {
          global_device_id_t const device =
              pt_mapping_get_device_for_coord(mapping, coord);
          DynamicTensorSlot const slot = DynamicTensorSlot{
              /*slot_name=*/base_slot.slot_name,
              /*slot_tensor_role=*/base_slot.slot_tensor_role,
              /*task_shard=*/
              needs_task_shard ? std::optional{device.coord} : std::nullopt,
          };
          return {slot,
                  DynamicValueAttrsShardingInfo{
                      /*shard_coord=*/coord,
                      /*mapping=*/device,
                  }};
        }));
  };

  // One sharding info per output coord (there is only 1 for GATHER)
  return transform(
      output_coords,
      [&](ParallelTensorSpaceCoordinate const &anchor)
          -> DynamicNodeInvocationShardingInfo {
        global_device_id_t const task_device =
            pt_mapping_get_device_for_coord(output_mapping, anchor);

        std::map<DynamicTensorSlot, DynamicValueAttrsShardingInfo> const
            input_sharding_infos =
                make_sharding_infos(input_coords, input_mapping, input_slot);
        std::map<DynamicTensorSlot, DynamicValueAttrsShardingInfo> const
            output_sharding_infos =
                make_sharding_infos({anchor}, output_mapping, output_slot);

        // device_ids must include every device that participates in this
        // invocation (all input-side devices plus the output device), not
        // just the output device — matching the SUM_REDUCE/BROADCAST cases,
        // which likewise include the full N-side device group. This matters
        // for consumers (e.g. machine slicing) that use device_ids to decide
        // which devices must execute/observe an invocation.
        std::set<global_device_id_t> device_id_set = transform(
            input_coords, [&](ParallelTensorSpaceCoordinate const &coord) {
              return pt_mapping_get_device_for_coord(input_mapping, coord);
            });
        device_id_set.insert(task_device);

        return DynamicNodeInvocationShardingInfo{
            /*device_ids=*/nonempty_set(device_id_set),
            /*value_sharding=*/
            binary_relation_from_map(binary_merge_disjoint_maps(
                input_sharding_infos, output_sharding_infos)),
        };
      });
}

std::set<DynamicNodeInvocation>
    perform_shard_expansion_for_invocation(DynamicNodeInvocation const &i) {

  std::set<DynamicNodeInvocationShardingInfo> shard_expansion_info =
      generate_shard_expansion_for_invocation(i);

  return transform(
      shard_expansion_info,
      [&](DynamicNodeInvocationShardingInfo const &s) -> DynamicNodeInvocation {
        return apply_dynamic_node_invocation_sharding_info(i, s);
      });
}

DynamicNodeAttrs apply_dynamic_node_attrs_sharding_info(
    DynamicNodeAttrs const &node_attrs,
    nonempty_set<global_device_id_t> const &device_ids) {
  DynamicNodeAttrs result = node_attrs;
  result.device_ids = device_ids;

  return result;
}

DynamicValueAttrs apply_dynamic_value_attrs_sharding_info(
    DynamicValueAttrs const &value_attrs,
    DynamicValueAttrsShardingInfo const &value_sharding_info) {
  DynamicValueAttrs result = value_attrs;
  result.shard_coord = value_sharding_info.shard_coord;

  if (result.mapping.has_value()) {
    ParallelTensorMapping value_mapping = assert_unwrap(result.mapping);

    global_device_id_t from_mapping = pt_mapping_get_device_for_coord(
        value_mapping, value_sharding_info.shard_coord);
    global_device_id_t from_sharding_info = value_sharding_info.mapping;

    ASSERT(from_mapping == from_sharding_info);
  }

  return result;
}

DynamicNodeInvocation apply_dynamic_node_invocation_sharding_info(
    DynamicNodeInvocation const &invocation,
    DynamicNodeInvocationShardingInfo const &invocation_sharding_info) {
  require_invocation_is_ready_for_shard_expansion(invocation);

  {
    std::set<DynamicTensorSlot> invocation_slots =
        set_union(keys(invocation.inputs), keys(invocation.outputs));

    std::set<DynamicTensorSlot> shard_info_slots_ignoring_task_shard =
        transform(invocation_sharding_info.value_sharding.left_values(),
                  slot_without_task_shard);

    ASSERT(invocation_slots == shard_info_slots_ignoring_task_shard,
           dynamic_node_invocation_to_serializable(invocation),
           invocation_sharding_info);
  }

  std::set<DynamicTensorSlot> shard_labelled = filtrans(
      invocation_sharding_info.value_sharding.left_values(),
      [](DynamicTensorSlot const &s) -> std::optional<DynamicTensorSlot> {
        if (s.task_shard.has_value()) {
          return slot_without_task_shard(s);
        } else {
          return std::nullopt;
        }
      });

  {
    std::set<DynamicTensorSlot> not_shard_labelled =
        filter(invocation_sharding_info.value_sharding.left_values(),
               [](DynamicTensorSlot const &s) -> bool {
                 return !s.task_shard.has_value();
               });

    ASSERT(are_disjoint(shard_labelled, not_shard_labelled));
  }

  auto shard_value = [&](DynamicTensorSlot const &slot,
                         DynamicValueAttrs const &value_attrs)
      -> std::map<DynamicTensorSlot, DynamicValueAttrs> {
    ASSERT(!slot.task_shard.has_value());

    if (contains(shard_labelled, slot)) {
      BinaryRelation<DynamicTensorSlot, DynamicValueAttrsShardingInfo>
          for_slot = filter_binary_relation(
              invocation_sharding_info.value_sharding,
              [&](DynamicTensorSlot const &s,
                  DynamicValueAttrsShardingInfo const &) -> bool {
                return slot_without_task_shard(s) == slot;
              });

      std::set<std::pair<DynamicTensorSlot, DynamicValueAttrs>> result =
          transform(for_slot.unwrap_as_set(),
                    [&](std::pair<DynamicTensorSlot,
                                  DynamicValueAttrsShardingInfo> const &p)
                        -> std::pair<DynamicTensorSlot, DynamicValueAttrs> {
                      return {p.first,
                              apply_dynamic_value_attrs_sharding_info(
                                  value_attrs, p.second)};
                    });

      return map_from_pairs(result);
    } else {
      DynamicValueAttrsShardingInfo sharding_info =
          get_only(invocation_sharding_info.value_sharding.at_l(slot));
      return {
          {
              slot,
              apply_dynamic_value_attrs_sharding_info(value_attrs,
                                                      sharding_info),
          },
      };
    }
  };

  DynamicNodeInvocation result = DynamicNodeInvocation{
      /*inputs=*/flatmap(invocation.inputs, shard_value),
      /*node_attrs=*/
      apply_dynamic_node_attrs_sharding_info(
          invocation.node_attrs, invocation_sharding_info.device_ids),
      /*outputs=*/flatmap(invocation.outputs, shard_value),
  };

  require_invocation_is_fully_shard_expanded(result);

  return result;
}

std::set<DynamicNodeInvocationShardingInfo>
    generate_shard_expansion_for_invocation(DynamicNodeInvocation const &i) {
  require_invocation_is_ready_for_shard_expansion(i);

  std::set<DynamicNodeInvocationShardingInfo> const result = [&]() {
    TrainingOperationAttrs const op_attrs =
        assert_unwrap(i.node_attrs.op_attrs);

    if (op_attrs.is_copy()) {
      return generate_shard_expansion_for_copy(i);
    }

    if (is_parallel_training_op(op_attrs)) {
      DynamicTaskType const task_type = assert_unwrap(i.node_attrs.task_type);
      ParallelOpMovementKind const kind =
          get_parallel_op_movement_kind_for_training_op(op_attrs, task_type)
              .value();
      return generate_shard_expansion_for_parallel_op(i, kind);
    }

    DynamicNodeMapping const mapping = assert_unwrap(i.node_attrs.mapping);
    std::set<global_device_id_t> const shard_machine_coords =
        target_devices_of_dynamic_node_mapping(mapping);

    return transform(shard_machine_coords,
                     [&](global_device_id_t const &device_id)
                         -> DynamicNodeInvocationShardingInfo {
                       OperatorAtomicTaskShardBinding const slot_bindings =
                           dynamic_node_mapping_get_shard_binding_for_device(
                               mapping, device_id);
                       return invocation_sharding_info_for_binding(
                           i, device_id, slot_bindings);
                     });
  }();

  return result;
}

DynamicOpenDataflowGraph
    perform_shard_expansion(DynamicOpenDataflowGraph const &g) {

  require_graph_is_ready_for_shard_expansion(g);

  DynamicOpenDataflowGraph result =
      flatmap_dynamic_invocation_set(g, [&](DynamicNodeInvocation const &i) {
        return perform_shard_expansion_for_invocation(i);
      });

  require_graph_is_fully_shard_expanded(result);

  return result;
}

} // namespace FlexFlow
