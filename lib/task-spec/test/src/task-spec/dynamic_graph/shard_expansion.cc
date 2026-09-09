#include "task-spec/dynamic_graph/shard_expansion.h"
#include "op-attrs/ops/element_unary.h"
#include "pcg/mapped_parallel_computation_graph/mapped_operator_task_group.h"
#include "task-spec/dynamic_graph/copy_attrs.dtg.h"
#include "task-spec/dynamic_graph/dynamic_copy_layer_guid_t.dtg.h"
#include "task-spec/dynamic_graph/dynamic_node_mapping.h"
#include "task-spec/dynamic_graph/dynamic_tensor_role.h"
#include "task-spec/dynamic_graph/serializable_dynamic_node_invocation.h"
#include "task-spec/dynamic_graph/training_operation_attrs.dtg.h"
#include "test/utils/doctest/check_kv.h"
#include "test/utils/doctest/fmt/set.h"
#include "utils/bidict/algorithms/bidict_filter_keys.h"
#include "utils/bidict/algorithms/bidict_filter_values.h"
#include "utils/binary_relation/binary_relation_from_map.h"
#include "utils/containers/binary_merge_disjoint_maps.h"
#include "utils/containers/map_from_pairs.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

static MachineSpaceCoordinate mk_machine_coord(nonnegative_int node_idx,
                                               nonnegative_int device_idx) {
  return MachineSpaceCoordinate{
      /*node_idx=*/node_idx,
      /*device_idx=*/device_idx,
  };
};

static global_device_id_t mk_device_id(MachineSpaceCoordinate const &mc) {
  return global_device_id_t{
      /*coord=*/mc,
      /*device_type=*/DeviceType::GPU,
  };
};

static ParallelTensorSpaceCoordinate mk_pt_coord(nonnegative_int idx1,
                                                 nonnegative_int idx2,
                                                 nonnegative_int idx3,
                                                 nonnegative_int idx4) {
  return ParallelTensorSpaceCoordinate{
      /*sum_component=*/idx1,
      /*discard_copy_component=*/idx2,
      /*shard_components=*/
      FFOrdered{
          idx3,
          idx4,
      },
  };
};

DynamicTensorSlot mk_slot(
    TensorSlotName const &slot_name,
    std::optional<MachineSpaceCoordinate> const &task_shard = std::nullopt) {
  return DynamicTensorSlot{
      /*slot_name=*/slot_name,
      /*slot_tensor_role=*/std::nullopt,
      /*task_shard=*/task_shard,
  };
};

DynamicValueAttrs
    mk_value(size_t src_node_id,
             TensorSlotName src_slot_name,
             bidict<ParallelTensorSpaceCoordinate, global_device_id_t> const
                 &tensor_binding,
             std::optional<ParallelTensorSpaceCoordinate> const &shard_coord,
             std::optional<DynamicTensorRole> const &role = std::nullopt) {

  bidict<ParallelTensorSpaceCoordinate, global_device_id_t> mapping =
      tensor_binding;
  if (shard_coord.has_value()) {
    mapping = bidict_filter_keys(mapping,
                                 [&](ParallelTensorSpaceCoordinate const &p) {
                                   return p == shard_coord.value();
                                 });
  }

  return DynamicValueAttrs{
      /*tensor_guid=*/dynamic_tensor_guid_t{parallel_tensor_guid_t{
          KwargDataflowOutput<TensorSlotName>{
              Node{src_node_id},
              src_slot_name,
          },
      }},
      /*parallel_tensor_shape=*/std::nullopt,
      /*create_grad=*/std::nullopt,
      /*subgradient_id=*/std::nullopt,
      /*shard_coord=*/shard_coord,
      /*mapping=*/ParallelTensorMapping{mapping},
      /*accessor=*/std::nullopt,
      /*role=*/role,
  };
};

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("apply_dynamic_node_invocation_sharding_info") {
    global_device_id_t device_0 = global_device_id_t{
        /*coord=*/MachineSpaceCoordinate{
            /*node_idx=*/0_n,
            /*device_idx=*/0_n,
        },
        /*device_type=*/DeviceType::GPU,
    };

    global_device_id_t device_1 = global_device_id_t{
        /*coord=*/MachineSpaceCoordinate{
            /*node_idx=*/2_n,
            /*device_idx=*/1_n,
        },
        /*device_type=*/DeviceType::GPU,
    };

    auto mk_slot = [](TensorSlotName slot_name,
                      std::optional<MachineSpaceCoordinate> const &task_shard =
                          std::nullopt) -> DynamicTensorSlot {
      return DynamicTensorSlot{
          /*slot_name=*/slot_name,
          /*slot_tensor_role=*/std::nullopt,
          /*task_shard=*/task_shard,
      };
    };

    auto mk_value =
        [](size_t src_node_id,
           TensorSlotName src_slot_name,
           std::optional<ParallelTensorSpaceCoordinate> const &shard_coord =
               std::nullopt) -> DynamicValueAttrs {
      return DynamicValueAttrs{
          /*tensor_guid=*/dynamic_tensor_guid_t{
              parallel_tensor_guid_t{
                  KwargDataflowOutput<TensorSlotName>{
                      /*node=*/Node{src_node_id},
                      /*slot_name=*/src_slot_name,
                  },
              },
          },
          /*parallel_tensor_shape=*/std::nullopt,
          /*create_grad=*/std::nullopt,
          /*subgradient_id=*/std::nullopt,
          /*shard_coord=*/shard_coord,
          /*mapping=*/std::nullopt,
          /*accessor=*/std::nullopt,
          /*role=*/std::nullopt,
      };
    };

    SUBCASE("sharding info creates additional arguments ie replicate") {
      auto mk_pt_coord =
          [](nonnegative_int idx) -> ParallelTensorSpaceCoordinate {
        return ParallelTensorSpaceCoordinate{
            /*sum_component=*/0_n,
            /*discard_copy_component=*/idx,
            /*shard_components=*/
            FFOrdered{
                0_n,
                0_n,
            },
        };
      };

      size_t input_src_node_id = 234;
      size_t replicate_layer_node_id = 13;

      DynamicNodeMapping node_mapping = DynamicNodeMapping{
          /*op_task_group=*/MappedOperatorTaskGroup{{
              {
                  device_0.coord,
                  OperatorAtomicTaskShardBinding{{
                      {
                          TensorSlotName::INPUT,
                          mk_pt_coord(0_n),
                      },
                      {
                          TensorSlotName::OUTPUT,
                          mk_pt_coord(0_n),
                      },
                  }},
              },
              {
                  device_1.coord,
                  OperatorAtomicTaskShardBinding{{
                      {
                          TensorSlotName::INPUT,
                          mk_pt_coord(0_n),
                      },
                      {
                          TensorSlotName::OUTPUT,
                          mk_pt_coord(1_n),
                      },
                  }},
              },
          }},
          /*device_type=*/DeviceType::GPU,
      };

      TrainingOperationAttrs op_attrs = TrainingOperationAttrs{
          PCGOperatorAttrs{
              ReplicateAttrs{
                  /*replicate_degree=*/2_p,
              },
          },
      };

      dynamic_layer_guid_t layer_guid = dynamic_layer_guid_t{
          parallel_layer_guid_t{
              Node{replicate_layer_node_id},
          },
      };

      DynamicNodeInvocation invocation = DynamicNodeInvocation{
          /*inputs=*/{
              {
                  mk_slot(TensorSlotName::INPUT),
                  mk_value(input_src_node_id, TensorSlotName::OUTPUT),
              },
          },
          /*node_attrs=*/
          DynamicNodeAttrs{
              /*task_type=*/std::nullopt,
              /*device_ids=*/std::nullopt,
              /*mapping=*/node_mapping,
              /*op_attrs=*/op_attrs,
              /*layer_guid=*/layer_guid,
              /*per_device_op_state=*/std::nullopt,
          },
          /*outputs=*/
          {
              {
                  mk_slot(TensorSlotName::OUTPUT),
                  mk_value(replicate_layer_node_id, TensorSlotName::OUTPUT),
              },
          },
      };

      DynamicNodeInvocationShardingInfo invocation_sharding_info =
          DynamicNodeInvocationShardingInfo{
              /*device_ids=*/nonempty_set{
                  device_0,
                  device_1,
              },
              /*value_sharding=*/
              {
                  {
                      mk_slot(TensorSlotName::INPUT),
                      DynamicValueAttrsShardingInfo{
                          /*shard_coord=*/mk_pt_coord(0_n),
                          /*mapping=*/device_0,
                      },
                  },
                  {
                      mk_slot(TensorSlotName::OUTPUT,
                              /*task_shard=*/device_0.coord),
                      DynamicValueAttrsShardingInfo{
                          /*shard_coord=*/mk_pt_coord(0_n),
                          /*mapping=*/device_0,
                      },
                  },
                  {
                      mk_slot(TensorSlotName::OUTPUT,
                              /*task_shard=*/device_1.coord),
                      DynamicValueAttrsShardingInfo{
                          /*shard_coord=*/mk_pt_coord(1_n),
                          /*mapping=*/device_1,
                      },
                  },
              },
          };

      DynamicNodeInvocation result =
          apply_dynamic_node_invocation_sharding_info(invocation,
                                                      invocation_sharding_info);

      DynamicNodeInvocation correct = DynamicNodeInvocation{
          /*inputs=*/{
              {
                  mk_slot(TensorSlotName::INPUT),
                  mk_value(input_src_node_id,
                           TensorSlotName::OUTPUT,
                           /*shrad_coord=*/mk_pt_coord(0_n)),
              },
          },
          /*node_attrs=*/
          DynamicNodeAttrs{
              /*task_type=*/std::nullopt,
              /*device_ids=*/
              nonempty_set{
                  device_0,
                  device_1,
              },
              /*mapping=*/node_mapping,
              /*op_attrs=*/op_attrs,
              /*layer_guid=*/layer_guid,
              /*per_device_op_state=*/std::nullopt,
          },
          /*outputs=*/
          {
              {
                  mk_slot(TensorSlotName::OUTPUT,
                          /*task_shard=*/device_0.coord),
                  mk_value(replicate_layer_node_id,
                           TensorSlotName::OUTPUT,
                           /*shard_coord=*/mk_pt_coord(0_n)),
              },
              {
                  mk_slot(TensorSlotName::OUTPUT,
                          /*task_shard=*/device_1.coord),
                  mk_value(replicate_layer_node_id,
                           TensorSlotName::OUTPUT,
                           /*shard_coord=*/mk_pt_coord(1_n)),
              },
          },
      };

      nlohmann::json result_json =
          dynamic_node_invocation_to_serializable(result);
      nlohmann::json correct_json =
          dynamic_node_invocation_to_serializable(correct);

      CHECK_MESSAGE(result == correct,
                    check_kv("result\n", result_json.dump()),
                    check_kv("correct\n", correct_json.dump()));
    }

    SUBCASE("sharding info does not create additional arguments ie standard "
            "operator") {
      size_t input_src_node_id = 234;
      size_t weight_src_node_id = 345;
      size_t linear_layer_node_id = 13;

      auto mk_pt_coord =
          [](nonnegative_int idx) -> ParallelTensorSpaceCoordinate {
        return ParallelTensorSpaceCoordinate{
            /*sum_component=*/0_n,
            /*discard_copy_component=*/idx,
            /*shard_components=*/
            FFOrdered{
                0_n,
                0_n,
            },
        };
      };

      // note that the node mapping does not have to be accurate/real here.
      // apply_dynamic_node_invocation_sharding_info should just function
      // based on what it is given.
      DynamicNodeMapping node_mapping = DynamicNodeMapping{
          /*op_task_group=*/MappedOperatorTaskGroup{{
              {
                  device_0.coord,
                  OperatorAtomicTaskShardBinding{{
                      {
                          TensorSlotName::INPUT,
                          mk_pt_coord(0_n),
                      },
                      {
                          TensorSlotName::WEIGHT,
                          mk_pt_coord(0_n),
                      },
                      {
                          TensorSlotName::OUTPUT,
                          mk_pt_coord(0_n),
                      },
                  }},
              },
              {
                  device_1.coord,
                  OperatorAtomicTaskShardBinding{{
                      {
                          TensorSlotName::INPUT,
                          mk_pt_coord(1_n),
                      },
                      {
                          TensorSlotName::WEIGHT,
                          mk_pt_coord(2_n),
                      },
                      {
                          TensorSlotName::OUTPUT,
                          mk_pt_coord(1_n),
                      },
                  }},
              },
          }},
          /*device_type=*/DeviceType::GPU,
      };

      TrainingOperationAttrs op_attrs = TrainingOperationAttrs{
          PCGOperatorAttrs{
              LinearAttrs{
                  /*out_channels=*/8_p,
                  /*use_bias=*/false,
                  /*data_type=*/DataType::FLOAT,
                  /*activation=*/std::nullopt,
                  /*regularizer=*/std::nullopt,
              },
          },
      };

      dynamic_layer_guid_t layer_guid = dynamic_layer_guid_t{
          parallel_layer_guid_t{
              Node{linear_layer_node_id},
          },
      };

      DynamicNodeInvocation invocation = DynamicNodeInvocation{
          /*inputs=*/{
              {
                  mk_slot(TensorSlotName::INPUT),
                  mk_value(input_src_node_id, TensorSlotName::OUTPUT),
              },
              {
                  mk_slot(TensorSlotName::WEIGHT),
                  mk_value(weight_src_node_id, TensorSlotName::OUTPUT),
              },
          },
          /*node_attrs=*/
          DynamicNodeAttrs{
              /*task_type=*/std::nullopt,
              /*device_ids=*/std::nullopt,
              /*mapping=*/node_mapping,
              /*op_attrs=*/op_attrs,
              /*layer_guid=*/layer_guid,
              /*per_device_op_state=*/std::nullopt,
          },
          /*outputs=*/
          {
              {
                  mk_slot(TensorSlotName::OUTPUT),
                  mk_value(linear_layer_node_id, TensorSlotName::OUTPUT),
              },
          },
      };

      DynamicNodeInvocationShardingInfo invocation_sharding_info =
          DynamicNodeInvocationShardingInfo{
              /*device_ids=*/nonempty_set{
                  device_1,
              },
              /*value_sharding=*/
              {
                  {
                      mk_slot(TensorSlotName::INPUT),
                      DynamicValueAttrsShardingInfo{
                          /*shard_coord=*/mk_pt_coord(1_n),
                          /*mapping=*/device_1,
                      },
                  },
                  {
                      mk_slot(TensorSlotName::WEIGHT),
                      DynamicValueAttrsShardingInfo{
                          /*shard_coord=*/mk_pt_coord(2_n),
                          /*mapping=*/device_1,
                      },
                  },
                  {
                      mk_slot(TensorSlotName::OUTPUT),
                      DynamicValueAttrsShardingInfo{
                          /*shard_coord=*/mk_pt_coord(1_n),
                          /*mapping=*/device_1,
                      },
                  },
              },
          };

      DynamicNodeInvocation result =
          apply_dynamic_node_invocation_sharding_info(invocation,
                                                      invocation_sharding_info);

      DynamicNodeInvocation correct = DynamicNodeInvocation{
          /*inputs=*/{
              {
                  mk_slot(TensorSlotName::INPUT),
                  mk_value(input_src_node_id,
                           TensorSlotName::OUTPUT,
                           /*shard_coord=*/mk_pt_coord(1_n)),
              },
              {
                  mk_slot(TensorSlotName::WEIGHT),
                  mk_value(weight_src_node_id,
                           TensorSlotName::OUTPUT,
                           /*shard_coord=*/mk_pt_coord(2_n)),
              },
          },
          /*node_attrs=*/
          DynamicNodeAttrs{
              /*task_type=*/std::nullopt,
              /*device_ids=*/nonempty_set{device_1},
              /*mapping=*/node_mapping,
              /*op_attrs=*/op_attrs,
              /*layer_guid=*/layer_guid,
              /*per_device_op_state=*/std::nullopt,
          },
          /*outputs=*/
          {
              {
                  mk_slot(TensorSlotName::OUTPUT),
                  mk_value(linear_layer_node_id,
                           TensorSlotName::OUTPUT,
                           /*shard_coord=*/mk_pt_coord(1_n)),
              },
          },
      };

      nlohmann::json result_json =
          dynamic_node_invocation_to_serializable(result);
      nlohmann::json correct_json =
          dynamic_node_invocation_to_serializable(correct);

      CHECK_MESSAGE(result == correct,
                    check_kv("result\n", result_json.dump()),
                    check_kv("correct\n", correct_json.dump()));
    }
  }

  TEST_CASE("generate_shard_expansion_for_invocation") {
    auto mk_op_value =
        [&](size_t src_node_id,
            TensorSlotName src_slot_name,
            TensorSlotName use_slot_name,
            DynamicNodeMapping const &node_mapping,
            std::optional<ParallelTensorSpaceCoordinate> const &shard_coord,
            std::optional<DynamicTensorRole> const &role =
                std::nullopt) -> DynamicValueAttrs {
      bidict<ParallelTensorSpaceCoordinate, global_device_id_t> tensor_binding =
          dynamic_node_mapping_bindings_for_slot_name(node_mapping,
                                                      use_slot_name);
      return mk_value(
          src_node_id, src_slot_name, tensor_binding, shard_coord, role);
    };

    auto mk_sharding_info =
        [&](TensorSlotName slot_name,
            ParallelTensorSpaceCoordinate const &shard_coord,
            DynamicNodeMapping const &node_mapping)
        -> std::pair<DynamicTensorSlot, DynamicValueAttrsShardingInfo> {
      bidict<ParallelTensorSpaceCoordinate, global_device_id_t> tensor_binding =
          dynamic_node_mapping_bindings_for_slot_name(node_mapping, slot_name);

      return std::pair{
          mk_slot(slot_name),
          DynamicValueAttrsShardingInfo{
              /*shard_coord=*/shard_coord,
              /*mapping=*/tensor_binding.at_l(shard_coord),
          },
      };
    };

    global_device_id_t dev1 = mk_device_id(mk_machine_coord(0_n, 0_n));
    global_device_id_t dev2 = mk_device_id(mk_machine_coord(1_n, 0_n));
    global_device_id_t dev3 = mk_device_id(mk_machine_coord(2_n, 0_n));
    global_device_id_t dev4 = mk_device_id(mk_machine_coord(3_n, 0_n));

    SUBCASE("standard operator") {
      auto mk_shard_binding = [&](ParallelTensorSpaceCoordinate const &c1,
                                  ParallelTensorSpaceCoordinate const &c2,
                                  ParallelTensorSpaceCoordinate const &c3,
                                  ParallelTensorSpaceCoordinate const &c4)
          -> OperatorAtomicTaskShardBinding {
        return OperatorAtomicTaskShardBinding{
            /*tensor_coords=*/{
                {
                    TensorSlotName::INPUT,
                    c1,
                },
                {
                    TensorSlotName::WEIGHT,
                    c2,
                },
                {
                    TensorSlotName::OUTPUT_01,
                    c3,
                },
                {
                    TensorSlotName::OUTPUT_02,
                    c4,
                },
            },
        };
      };

      auto mk_value =
          [&](size_t src_node_id,
              TensorSlotName src_slot_name,
              bidict<ParallelTensorSpaceCoordinate, global_device_id_t>
                  tensor_binding,
              std::optional<ParallelTensorSpaceCoordinate> const &shard_coord)
          -> DynamicValueAttrs {
        if (shard_coord.has_value()) {
          tensor_binding = bidict_filter_keys(
              tensor_binding,
              [&](ParallelTensorSpaceCoordinate const &p) -> bool {
                return p == shard_coord.value();
              });
        }

        return DynamicValueAttrs{
            /*tensor_guid=*/dynamic_tensor_guid_t{parallel_tensor_guid_t{
                KwargDataflowOutput<TensorSlotName>{
                    Node{src_node_id},
                    src_slot_name,
                },
            }},
            /*parallel_tensor_shape=*/std::nullopt,
            /*create_grad=*/std::nullopt,
            /*subgradient_id=*/std::nullopt,
            /*shard_coord=*/shard_coord,
            /*mapping=*/
            ParallelTensorMapping{tensor_binding},
            /*accessor=*/std::nullopt,
            /*role=*/std::nullopt,
        };
      };

      ParallelTensorSpaceCoordinate mc1_input_coord =
          mk_pt_coord(0_n, 0_n, 0_n, 0_n);
      ParallelTensorSpaceCoordinate mc1_weight_coord =
          mk_pt_coord(0_n, 1_n, 2_n, 0_n);
      ParallelTensorSpaceCoordinate mc1_output_1_coord =
          mk_pt_coord(1_n, 0_n, 0_n, 1_n);
      ParallelTensorSpaceCoordinate mc1_output_2_coord =
          mk_pt_coord(3_n, 0_n, 0_n, 0_n);

      ParallelTensorSpaceCoordinate mc2_input_coord =
          mk_pt_coord(0_n, 1_n, 0_n, 0_n);
      ParallelTensorSpaceCoordinate mc2_weight_coord =
          mk_pt_coord(0_n, 4_n, 2_n, 0_n);
      ParallelTensorSpaceCoordinate mc2_output_1_coord =
          mk_pt_coord(1_n, 2_n, 0_n, 1_n);
      ParallelTensorSpaceCoordinate mc2_output_2_coord =
          mk_pt_coord(0_n, 0_n, 0_n, 0_n);

      TrainingOperationAttrs op_attrs = TrainingOperationAttrs{
          PCGOperatorAttrs{
              make_relu_attrs(),
          },
      };

      DynamicNodeMapping node_mapping = DynamicNodeMapping{
          MappedOperatorTaskGroup{
              bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding>{
                  {
                      dev1.coord,
                      mk_shard_binding(mc1_input_coord,
                                       mc1_weight_coord,
                                       mc1_output_1_coord,
                                       mc1_output_2_coord),
                  },
                  {
                      dev2.coord,
                      mk_shard_binding(mc2_input_coord,
                                       mc2_weight_coord,
                                       mc2_output_1_coord,
                                       mc2_output_2_coord),
                  },
              },
          },
          DeviceType::GPU,
      };

      DynamicNodeInvocation input = DynamicNodeInvocation{
          /*inputs=*/{
              {
                  mk_slot(TensorSlotName::INPUT),
                  mk_op_value(0,
                              TensorSlotName::OUTPUT,
                              TensorSlotName::INPUT,
                              node_mapping,
                              std::nullopt),
              },
              {
                  mk_slot(TensorSlotName::WEIGHT),
                  mk_op_value(1,
                              TensorSlotName::OUTPUT,
                              TensorSlotName::WEIGHT,
                              node_mapping,
                              std::nullopt),
              },
          },
          /*node_attrs=*/
          DynamicNodeAttrs{
              /*task_type=*/std::nullopt,
              /*device_coord=*/std::nullopt,
              /*mapping=*/node_mapping,
              /*op_attrs=*/op_attrs,
              /*layer_guid=*/
              dynamic_layer_guid_t{parallel_layer_guid_t{Node{20}}},
              /*per_device_op_state=*/std::nullopt,
          },
          /*outputs=*/
          {
              {
                  mk_slot(TensorSlotName::OUTPUT_01),
                  mk_op_value(20,
                              TensorSlotName::OUTPUT_01,
                              TensorSlotName::OUTPUT_01,
                              node_mapping,
                              std::nullopt),
              },
              {
                  mk_slot(TensorSlotName::OUTPUT_02),
                  mk_op_value(20,
                              TensorSlotName::OUTPUT_02,
                              TensorSlotName::OUTPUT_02,
                              node_mapping,
                              std::nullopt),
              },
          },
      };

      std::set<DynamicNodeInvocationShardingInfo> result =
          generate_shard_expansion_for_invocation(input);

      auto mk_invocation_shard =
          [&](global_device_id_t const &device_coord,
              ParallelTensorSpaceCoordinate const &input_shard_coord,
              ParallelTensorSpaceCoordinate const &weight_shard_coord,
              ParallelTensorSpaceCoordinate const &output_1_shard_coord,
              ParallelTensorSpaceCoordinate const &output_2_shard_coord)
          -> DynamicNodeInvocationShardingInfo {
        return DynamicNodeInvocationShardingInfo{
            /*device_coord=*/nonempty_set{device_coord},
            /*value_sharding=*/
            {
                mk_sharding_info(
                    TensorSlotName::INPUT, input_shard_coord, node_mapping),
                mk_sharding_info(
                    TensorSlotName::WEIGHT, weight_shard_coord, node_mapping),
                mk_sharding_info(TensorSlotName::OUTPUT_01,
                                 output_1_shard_coord,
                                 node_mapping),
                mk_sharding_info(TensorSlotName::OUTPUT_02,
                                 output_2_shard_coord,
                                 node_mapping),
            },
        };
      };

      std::set<DynamicNodeInvocationShardingInfo> correct = {
          mk_invocation_shard(dev1,
                              mc1_input_coord,
                              mc1_weight_coord,
                              mc1_output_1_coord,
                              mc1_output_2_coord),
          mk_invocation_shard(dev2,
                              mc2_input_coord,
                              mc2_weight_coord,
                              mc2_output_1_coord,
                              mc2_output_2_coord),
      };

      CHECK(result.size() == correct.size());
      CHECK(result == correct);
    }

    SUBCASE("for copy operator") {
      ParallelTensorSpaceCoordinate pt1 = mk_pt_coord(0_n, 0_n, 0_n, 0_n);
      ParallelTensorSpaceCoordinate pt2 = mk_pt_coord(0_n, 1_n, 0_n, 0_n);

      bidict<ParallelTensorSpaceCoordinate, global_device_id_t> src_binding{
          {pt1, dev1},
          {pt2, dev2},
      };
      bidict<ParallelTensorSpaceCoordinate, global_device_id_t> dst_binding{
          {pt1, dev3},
          {pt2, dev4},
      };

      DynamicNodeInvocation input = DynamicNodeInvocation{
          /*inputs=*/{
              {
                  mk_slot(TensorSlotName::INPUT),
                  mk_value(
                      0, TensorSlotName::OUTPUT, src_binding, std::nullopt),
              },
          },
          /*node_attrs=*/
          DynamicNodeAttrs{
              /*task_type=*/std::nullopt,
              /*device_coord=*/std::nullopt,
              /*mapping=*/std::nullopt,
              /*op_attrs=*/TrainingOperationAttrs{CopyAttrs{}},
              /*layer_guid=*/dynamic_layer_guid_t{dynamic_copy_layer_guid_t{}},
              /*per_device_op_state=*/std::nullopt,
          },
          /*outputs=*/
          {
              {
                  mk_slot(TensorSlotName::OUTPUT),
                  mk_value(
                      20, TensorSlotName::OUTPUT, dst_binding, std::nullopt),
              },
          },
      };

      std::set<DynamicNodeInvocationShardingInfo> result =
          generate_shard_expansion_for_invocation(input);

      auto mk_invocation_shard =
          [&](global_device_id_t const &device_coord,
              ParallelTensorSpaceCoordinate const &tensor_shard_coord)
          -> DynamicNodeInvocationShardingInfo {
        return DynamicNodeInvocationShardingInfo{
            /*device_coord=*/nonempty_set{device_coord},
            /*value_sharding=*/
            BinaryRelation<DynamicTensorSlot, DynamicValueAttrsShardingInfo>{
                {
                    mk_slot(TensorSlotName::INPUT),
                    DynamicValueAttrsShardingInfo{
                        tensor_shard_coord,
                        src_binding.at_l(tensor_shard_coord),
                    },
                },
                {
                    mk_slot(TensorSlotName::OUTPUT),
                    DynamicValueAttrsShardingInfo{
                        tensor_shard_coord,
                        dst_binding.at_l(tensor_shard_coord),
                    },
                },
            },
        };
      };

      std::set<DynamicNodeInvocationShardingInfo> correct = {
          mk_invocation_shard(dev1, pt1),
          mk_invocation_shard(dev2, pt2),
      };

      CHECK(result.size() == correct.size());
      CHECK(result == correct);
    }

    SUBCASE("replicate operator") {
      ParallelTensorSpaceCoordinate pt1 = mk_pt_coord(0_n, 0_n, 0_n, 0_n);
      ParallelTensorSpaceCoordinate pt2 = mk_pt_coord(0_n, 0_n, 0_n, 1_n);
      ParallelTensorSpaceCoordinate pt3 = mk_pt_coord(0_n, 1_n, 0_n, 0_n);
      ParallelTensorSpaceCoordinate pt4 = mk_pt_coord(0_n, 1_n, 0_n, 1_n);

      auto mk_shard_binding = [&](ParallelTensorSpaceCoordinate const &c1,
                                  ParallelTensorSpaceCoordinate const &c2)
          -> OperatorAtomicTaskShardBinding {
        return OperatorAtomicTaskShardBinding{
            /*tensor_coords=*/{
                {
                    TensorSlotName::INPUT,
                    c1,
                },
                {
                    TensorSlotName::OUTPUT,
                    c2,
                },
            },
        };
      };

      DynamicNodeMapping node_mapping = DynamicNodeMapping{
          /*op_task_group=*/MappedOperatorTaskGroup{
              bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding>{
                  {
                      dev1.coord,
                      mk_shard_binding(pt1, pt1),
                  },
                  {
                      dev2.coord,
                      mk_shard_binding(pt1, pt2),
                  },
                  {
                      dev3.coord,
                      mk_shard_binding(pt2, pt3),
                  },
                  {
                      dev4.coord,
                      mk_shard_binding(pt2, pt4),
                  },
              },
          },
          /*device_type=*/DeviceType::GPU,
      };

      SUBCASE("fwd") {
        bidict<ParallelTensorSpaceCoordinate, global_device_id_t> src_binding{
            {pt1, dev1},
            {pt2, dev2},
        };

        bidict<ParallelTensorSpaceCoordinate, global_device_id_t> dst_binding{
            {pt1, dev1},
            {pt2, dev2},
            {pt3, dev3},
            {pt4, dev4},
        };

        DynamicNodeInvocation input = DynamicNodeInvocation{
            /*inputs=*/{
                {
                    DynamicTensorSlot{
                        /*slot_name=*/TensorSlotName::INPUT,
                        /*slot_tensor_role=*/mk_dynamic_tensor_role_fwd(),
                        /*task_shard=*/std::nullopt,
                    },
                    mk_value(
                        0, TensorSlotName::OUTPUT, src_binding, std::nullopt),
                },
            },
            /*node_attrs=*/
            DynamicNodeAttrs{
                /*task_type=*/DynamicTaskType::FWD,
                /*device_ids=*/std::nullopt,
                /*mapping=*/node_mapping,
                /*op_attrs=*/
                TrainingOperationAttrs{
                    PCGOperatorAttrs{
                        ReplicateAttrs{
                            /*replicate_degree=*/2_p,
                        },
                    },
                },
                /*layer_guid=*/
                dynamic_layer_guid_t{parallel_layer_guid_t{Node{20}}},
                /*per_device_op_state=*/std::nullopt,
            },
            /*outputs=*/
            {
                {
                    DynamicTensorSlot{
                        /*slot_name=*/TensorSlotName::OUTPUT,
                        /*slot_tensor_role=*/mk_dynamic_tensor_role_fwd(),
                        /*task_shard=*/std::nullopt,
                    },
                    mk_value(
                        20, TensorSlotName::OUTPUT, dst_binding, std::nullopt),
                },
            },
        };

        std::set<DynamicNodeInvocationShardingInfo> result =
            generate_shard_expansion_for_invocation(input);

        auto mk_output_binding = [&](global_device_id_t const &device)
            -> std::pair<DynamicTensorSlot, DynamicValueAttrsShardingInfo> {
          return {
              DynamicTensorSlot{
                  /*slot_name=*/TensorSlotName::OUTPUT,
                  /*slot_tensor_role=*/mk_dynamic_tensor_role_fwd(),
                  /*task_shard=*/device.coord,
              },
              DynamicValueAttrsShardingInfo{
                  dst_binding.at_r(device),
                  device,
              },
          };
        };

        auto mk_invocation_shard =
            [&](nonempty_set<global_device_id_t> const &device_ids,
                ParallelTensorSpaceCoordinate const &input_shard_coord,
                std::set<global_device_id_t> const &output_task_shards)
            -> DynamicNodeInvocationShardingInfo {
          return DynamicNodeInvocationShardingInfo{
              /*device_ids=*/device_ids,
              /*value_sharding=*/
              binary_relation_from_map(binary_merge_disjoint_maps(
                  std::map<DynamicTensorSlot, DynamicValueAttrsShardingInfo>{
                      {
                          DynamicTensorSlot{
                              /*slot_name=*/TensorSlotName::INPUT,
                              /*slot_tensor_role=*/mk_dynamic_tensor_role_fwd(),
                              /*task_shard=*/std::nullopt,
                          },
                          DynamicValueAttrsShardingInfo{
                              input_shard_coord,
                              src_binding.at_l(input_shard_coord),
                          },
                      },
                  },
                  map_from_pairs(
                      transform(output_task_shards, mk_output_binding)))),
          };
        };

        std::set<DynamicNodeInvocationShardingInfo> correct = {
            mk_invocation_shard(nonempty_set{dev1, dev2}, pt1, {dev1, dev2}),
            mk_invocation_shard(nonempty_set{dev3, dev4}, pt2, {dev3, dev4}),
        };

        CHECK(result.size() == correct.size());
        CHECK(result == correct);
      }

      SUBCASE("bwd") {
        bidict<ParallelTensorSpaceCoordinate, global_device_id_t>
            output_grad_binding{
                {pt1, dev1},
                {pt2, dev2},
                {pt3, dev3},
                {pt4, dev4},
            };

        bidict<ParallelTensorSpaceCoordinate, global_device_id_t>
            input_grad_binding{
                {pt1, dev1},
                {pt2, dev2},
            };

        DynamicNodeInvocation input = DynamicNodeInvocation{
            /*inputs=*/{
                {
                    DynamicTensorSlot{
                        /*slot_name=*/TensorSlotName::OUTPUT,
                        /*slot_tensor_role=*/mk_dynamic_tensor_role_bwd(),
                        /*task_shard=*/std::nullopt,
                    },
                    mk_value(0,
                             TensorSlotName::OUTPUT,
                             output_grad_binding,
                             std::nullopt),
                },
            },
            /*node_attrs=*/
            DynamicNodeAttrs{
                /*task_type=*/DynamicTaskType::BWD,
                /*device_ids=*/std::nullopt,
                /*mapping=*/node_mapping,
                /*op_attrs=*/
                TrainingOperationAttrs{
                    PCGOperatorAttrs{
                        ReplicateAttrs{
                            /*replicate_degree=*/2_p,
                        },
                    },
                },
                /*layer_guid=*/
                dynamic_layer_guid_t{parallel_layer_guid_t{Node{20}}},
                /*per_device_op_state=*/std::nullopt,
            },
            /*outputs=*/
            {
                {
                    DynamicTensorSlot{
                        /*slot_name=*/TensorSlotName::INPUT,
                        /*slot_tensor_role=*/mk_dynamic_tensor_role_bwd(),
                        /*task_shard=*/std::nullopt,
                    },
                    mk_value(20,
                             TensorSlotName::INPUT,
                             input_grad_binding,
                             std::nullopt),
                },
            },
        };

        std::set<DynamicNodeInvocationShardingInfo> result =
            generate_shard_expansion_for_invocation(input);

        auto mk_output_grad_binding = [&](global_device_id_t const &device)
            -> std::pair<DynamicTensorSlot, DynamicValueAttrsShardingInfo> {
          return {
              DynamicTensorSlot{
                  /*slot_name=*/TensorSlotName::OUTPUT,
                  /*slot_tensor_role=*/mk_dynamic_tensor_role_bwd(),
                  /*task_shard=*/device.coord,
              },
              DynamicValueAttrsShardingInfo{
                  output_grad_binding.at_r(device),
                  device,
              },
          };
        };

        auto mk_invocation_shard =
            [&](nonempty_set<global_device_id_t> const &device_ids,
                std::set<global_device_id_t> const &output_grad_task_shards,
                ParallelTensorSpaceCoordinate const &input_grad_shard_coord)
            -> DynamicNodeInvocationShardingInfo {
          return DynamicNodeInvocationShardingInfo{
              /*device_ids=*/device_ids,
              /*value_sharding=*/
              binary_relation_from_map(binary_merge_disjoint_maps(
                  std::map<DynamicTensorSlot, DynamicValueAttrsShardingInfo>{
                      {
                          DynamicTensorSlot{
                              /*slot_name=*/TensorSlotName::INPUT,
                              /*slot_tensor_role=*/mk_dynamic_tensor_role_bwd(),
                              /*task_shard=*/std::nullopt,
                          },
                          DynamicValueAttrsShardingInfo{
                              input_grad_shard_coord,
                              input_grad_binding.at_l(input_grad_shard_coord),
                          },
                      },
                  },
                  map_from_pairs(transform(output_grad_task_shards,
                                           mk_output_grad_binding)))),
          };
        };

        std::set<DynamicNodeInvocationShardingInfo> correct = {
            mk_invocation_shard(nonempty_set{dev1, dev2}, {dev1, dev2}, pt1),
            mk_invocation_shard(nonempty_set{dev3, dev4}, {dev3, dev4}, pt2),
        };

        CHECK(result.size() == correct.size());
        CHECK(result == correct);
      }
    }
  }
}
