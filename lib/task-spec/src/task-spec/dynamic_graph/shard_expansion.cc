#include "task-spec/dynamic_graph/shard_expansion.h"
#include "pcg/mapped_parallel_computation_graph/operator_atomic_task_shard_binding.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.h"
#include "task-spec/dynamic_graph/dynamic_node_mapping.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_tensor_role.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "task-spec/dynamic_graph/parallel_tensor_mapping.h"
#include "task-spec/dynamic_graph/serializable_dynamic_node_invocation.h"
#include "task-spec/dynamic_graph/shard_expansion.h"
#include "task-spec/dynamic_graph/training_operation_attrs.h"
#include "utils/bidict/algorithms/bidict_filter_keys.h"
#include "utils/bidict/algorithms/bidict_filter_values.h"
#include "utils/binary_relation/binary_relation_from_map.h"
#include "utils/binary_relation/binary_relation_transform_right2.h"
#include "utils/binary_relation/filter_binary_relation.h"
#include "utils/containers/are_disjoint.h"
#include "utils/containers/binary_merge_disjoint_maps.h"
#include "utils/containers/flatmap.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/get_only.h"
#include "utils/containers/map_from_pairs.h"
#include "utils/containers/map_from_unordered.h"
#include "utils/containers/map_keys.h"
#include "utils/containers/map_values2.h"
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

static bidict<ParallelTensorSpaceCoordinate, global_device_id_t>
    restrict_tensor_mapping_keys_to_coord(
        bidict<ParallelTensorSpaceCoordinate, global_device_id_t> const
            &mapping,
        ParallelTensorSpaceCoordinate const &parallel_tensor_coord) {
  return bidict_filter_keys(mapping,
                            [&](ParallelTensorSpaceCoordinate const &p) {
                              return p == parallel_tensor_coord;
                            });
}

static DynamicNodeInvocation shard_invocation_for_binding(
    DynamicNodeInvocation const &i,
    global_device_id_t const &device_id,
    OperatorAtomicTaskShardBinding const &binding) {

  auto shard_expand_value_attrs =
      [&](DynamicTensorSlot const &s,
          DynamicValueAttrs const &v) -> DynamicValueAttrs {
    ParallelTensorSpaceCoordinate parallel_tensor_coord =
        binding.tensor_coords.at(s.slot_name);

    DynamicValueAttrs result = v;
    result.shard_coord = parallel_tensor_coord;
    result.mapping = transform(
        v.mapping,
        [&](ParallelTensorMapping const &mapping) -> ParallelTensorMapping {
          return ParallelTensorMapping{
              restrict_tensor_mapping_keys_to_coord(mapping.raw,
                                                    parallel_tensor_coord),
          };
        });
    return result;
  };

  DynamicNodeAttrs expanded_node_attrs = [&]() {
    DynamicNodeAttrs result = i.node_attrs;
    result.device_ids = nonempty_set{device_id};
    ;
    return result;
  }();

  return DynamicNodeInvocation{
      /*inputs=*/map_values2(i.inputs, shard_expand_value_attrs),
      /*node_attrs=*/expanded_node_attrs,
      /*outputs=*/map_values2(i.outputs, shard_expand_value_attrs),
  };
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

// TODO(@lockshaw): There is a lot of code duplication between
// generate_shard_expansion_for_fwd_replicate and
// generate_shard_expansion_for_bwd_replicate that should eventually be
// factored out.
static std::set<DynamicNodeInvocationShardingInfo>
    generate_shard_expansion_for_fwd_replicate(DynamicNodeInvocation const &i) {
  ASSERT(i.node_attrs.task_type == DynamicTaskType::FWD);

  DynamicNodeMapping node_mapping = assert_unwrap(i.node_attrs.mapping);

  DynamicTensorSlot expected_input_slot = DynamicTensorSlot{
      /*slot_name=*/TensorSlotName::INPUT,
      /*slot_tensor_role=*/mk_dynamic_tensor_role_fwd(),
      /*task_shard=*/std::nullopt,
  };

  DynamicValueAttrs input = require_only_key(i.inputs, expected_input_slot);

  DynamicTensorSlot expected_output_slot = DynamicTensorSlot{
      /*slot_name=*/TensorSlotName::OUTPUT,
      /*slot_tensor_role=*/mk_dynamic_tensor_role_fwd(),
      /*task_shard=*/std::nullopt,
  };

  DynamicValueAttrs output = require_only_key(i.outputs, expected_output_slot);

  ParallelTensorMapping input_value_mapping = assert_unwrap(input.mapping);

  std::set<ParallelTensorSpaceCoordinate> input_tensor_shards =
      pt_mapping_get_coord_set(input_value_mapping);

  ParallelTensorMapping output_value_mapping = assert_unwrap(output.mapping);

  auto get_task_shard_device_ids_for_input_tensor_shard =
      [&](ParallelTensorSpaceCoordinate const &input_tensor_shard)
      -> nonempty_set<global_device_id_t> {
    bidict<global_device_id_t, OperatorAtomicTaskShardBinding>
        dependent_on_input_tensor_shard = bidict_filter_values(
            dynamic_node_mapping_get_shard_bindings(node_mapping),
            [&](OperatorAtomicTaskShardBinding const &b) -> bool {
              return ptensor_space_coord_for_slot_name(
                         b, TensorSlotName::INPUT) == input_tensor_shard;
            });

    return nonempty_set(dependent_on_input_tensor_shard.left_values());
  };

  auto invocation_sharding_info_for_input_tensor_shard =
      [&](ParallelTensorSpaceCoordinate const &c)
      -> DynamicNodeInvocationShardingInfo {
    nonempty_set<global_device_id_t> task_shard_device_ids =
        get_task_shard_device_ids_for_input_tensor_shard(c);

    std::map<global_device_id_t, DynamicValueAttrsShardingInfo>
        output_sharding_infos =
            generate_map(task_shard_device_ids.unwrap_as_set(),
                         [&](global_device_id_t const &device_id)
                             -> DynamicValueAttrsShardingInfo {
                           ParallelTensorSpaceCoordinate pc =
                               pt_mapping_get_coord_for_device(
                                   output_value_mapping, device_id);

                           return DynamicValueAttrsShardingInfo{
                               /*shard_coord=*/pc,
                               /*mapping=*/device_id,
                           };
                         });

    std::map<DynamicTensorSlot, DynamicValueAttrsShardingInfo>
        keyed_output_sharding_infos = map_keys(
            output_sharding_infos,
            [&](global_device_id_t const &device_id) -> DynamicTensorSlot {
              return DynamicTensorSlot{
                  /*slot_name=*/TensorSlotName::OUTPUT,
                  /*slot_tensor_role=*/mk_dynamic_tensor_role_fwd(),
                  /*task_shard=*/device_id.coord,
              };
            });

    DynamicTensorSlot input_slot = DynamicTensorSlot{
        /*slot_name=*/TensorSlotName::INPUT,
        /*slot_tensor_role=*/mk_dynamic_tensor_role_fwd(),
        /*task_shard=*/std::nullopt,
    };

    DynamicValueAttrsShardingInfo input_sharding_info =
        DynamicValueAttrsShardingInfo{
            /*shard_coord=*/c,
            /*mapping=*/pt_mapping_get_device_for_coord(input_value_mapping, c),
        };

    std::map<DynamicTensorSlot, DynamicValueAttrsShardingInfo> sharding_infos =
        binary_merge_disjoint_maps(
            keyed_output_sharding_infos,
            std::map<DynamicTensorSlot, DynamicValueAttrsShardingInfo>{
                {
                    input_slot,
                    input_sharding_info,
                },
            });

    return DynamicNodeInvocationShardingInfo{
        /*device_ids=*/task_shard_device_ids,
        /*value_sharding=*/binary_relation_from_map(sharding_infos),
    };
  };

  return transform(input_tensor_shards,
                   invocation_sharding_info_for_input_tensor_shard);
}

static std::set<DynamicNodeInvocationShardingInfo>
    generate_shard_expansion_for_bwd_replicate(DynamicNodeInvocation const &i) {
  ASSERT(i.node_attrs.task_type == DynamicTaskType::BWD);

  DynamicNodeMapping node_mapping = assert_unwrap(i.node_attrs.mapping);

  DynamicTensorSlot expected_output_grad_slot = DynamicTensorSlot{
      /*slot_name=*/TensorSlotName::OUTPUT,
      /*slot_tensor_role=*/mk_dynamic_tensor_role_bwd(),
      /*task_shard=*/std::nullopt,
  };

  DynamicValueAttrs output_grad =
      require_only_key(i.inputs, expected_output_grad_slot);

  DynamicTensorSlot expected_input_grad_slot = DynamicTensorSlot{
      /*slot_name=*/TensorSlotName::INPUT,
      /*slot_tensor_role=*/mk_dynamic_tensor_role_bwd(),
      /*task_shard=*/std::nullopt,
  };

  DynamicValueAttrs input_grad =
      require_only_key(i.outputs, expected_input_grad_slot);

  ParallelTensorMapping output_grad_value_mapping =
      assert_unwrap(output_grad.mapping);
  ParallelTensorMapping input_grad_value_mapping =
      assert_unwrap(input_grad.mapping);

  std::set<ParallelTensorSpaceCoordinate> input_grad_tensor_shards =
      pt_mapping_get_coord_set(input_grad_value_mapping);

  auto get_task_shard_device_ids_for_input_grad_tensor_shard =
      [&](ParallelTensorSpaceCoordinate const &input_grad_tensor_shard)
      -> nonempty_set<global_device_id_t> {
    bidict<global_device_id_t, OperatorAtomicTaskShardBinding>
        produce_input_grad_tensor_shard = bidict_filter_values(
            dynamic_node_mapping_get_shard_bindings(node_mapping),
            [&](OperatorAtomicTaskShardBinding const &b) -> bool {
              return ptensor_space_coord_for_slot_name(
                         b, TensorSlotName::INPUT) == input_grad_tensor_shard;
            });

    return nonempty_set(produce_input_grad_tensor_shard.left_values());
  };

  auto invocation_sharding_info_for_input_grad_tensor_shard =
      [&](ParallelTensorSpaceCoordinate const &c)
      -> DynamicNodeInvocationShardingInfo {
    nonempty_set<global_device_id_t> task_shard_device_ids =
        get_task_shard_device_ids_for_input_grad_tensor_shard(c);

    std::map<global_device_id_t, DynamicValueAttrsShardingInfo>
        output_grad_sharding_infos =
            generate_map(task_shard_device_ids.unwrap_as_set(),
                         [&](global_device_id_t const &device_id)
                             -> DynamicValueAttrsShardingInfo {
                           ParallelTensorSpaceCoordinate pc =
                               pt_mapping_get_coord_for_device(
                                   output_grad_value_mapping, device_id);

                           return DynamicValueAttrsShardingInfo{
                               /*shard_coord=*/pc,
                               /*mapping=*/device_id,
                           };
                         });

    std::map<DynamicTensorSlot, DynamicValueAttrsShardingInfo>
        keyed_output_grad_sharding_infos = map_keys(
            output_grad_sharding_infos,
            [&](global_device_id_t const &device_id) -> DynamicTensorSlot {
              return DynamicTensorSlot{
                  /*slot_name=*/TensorSlotName::OUTPUT,
                  /*slot_tensor_role=*/mk_dynamic_tensor_role_bwd(),
                  /*task_shard=*/device_id.coord,
              };
            });

    DynamicTensorSlot input_grad_slot = DynamicTensorSlot{
        /*slot_name=*/TensorSlotName::INPUT,
        /*slot_tensor_role=*/mk_dynamic_tensor_role_bwd(),
        /*task_shard=*/std::nullopt,
    };

    DynamicValueAttrsShardingInfo input_grad_sharding_info =
        DynamicValueAttrsShardingInfo{
            /*shard_coord=*/c,
            /*mapping=*/
            pt_mapping_get_device_for_coord(input_grad_value_mapping, c),
        };

    std::map<DynamicTensorSlot, DynamicValueAttrsShardingInfo> sharding_infos =
        binary_merge_disjoint_maps(
            keyed_output_grad_sharding_infos,
            std::map<DynamicTensorSlot, DynamicValueAttrsShardingInfo>{
                {
                    input_grad_slot,
                    input_grad_sharding_info,
                },
            });

    return DynamicNodeInvocationShardingInfo{
        /*device_ids=*/task_shard_device_ids,
        /*value_sharding=*/binary_relation_from_map(sharding_infos),
    };
  };

  return transform(input_grad_tensor_shards,
                   invocation_sharding_info_for_input_grad_tensor_shard);
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

  std::set<DynamicNodeInvocationShardingInfo> result = [&]() {
    if (i.node_attrs.op_attrs.value().is_copy()) {
      return generate_shard_expansion_for_copy(i);
    } else if (training_op_attrs_has_op_type(i.node_attrs.op_attrs.value(),
                                             OperatorType::REPLICATE)) {
      DynamicTaskType task_type = assert_unwrap(i.node_attrs.task_type);
      switch (task_type) {
        case DynamicTaskType::FWD:
          return set_of(generate_shard_expansion_for_fwd_replicate(i));
        case DynamicTaskType::BWD:
          return set_of(generate_shard_expansion_for_bwd_replicate(i));
        default:
          PANIC("Unexpected task type for Replicate: {}", task_type);
      }
    } else {
      DynamicNodeMapping mapping = assert_unwrap(i.node_attrs.mapping);

      std::set<global_device_id_t> shard_machine_coords =
          target_devices_of_dynamic_node_mapping(mapping);

      return transform(shard_machine_coords,
                       [&](global_device_id_t const &device_id)
                           -> DynamicNodeInvocationShardingInfo {
                         OperatorAtomicTaskShardBinding slot_bindings =
                             dynamic_node_mapping_get_shard_binding_for_device(
                                 mapping, device_id);

                         return invocation_sharding_info_for_binding(
                             i, device_id, slot_bindings);
                       });
    }
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
