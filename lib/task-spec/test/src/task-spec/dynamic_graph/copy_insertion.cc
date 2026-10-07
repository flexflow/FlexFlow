#include "task-spec/dynamic_graph/copy_insertion.h"
#include "op-attrs/ops/element_unary.h"
#include "op-attrs/tensor_slot_name.dtg.h"
#include "pcg/mapped_parallel_computation_graph/mapped_operator_task_group.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_task_type.dtg.h"
#include "task-spec/dynamic_graph/dynamic_tensor_role.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.h"
#include "task-spec/dynamic_graph/pass_expansion.h"
#include "task-spec/dynamic_graph/serializable_dynamic_node_invocation.h"
#include "task-spec/dynamic_graph/serializable_dynamic_open_dataflow_graph.h"
#include "test/utils/doctest/check_kv.h"
#include "test/utils/doctest/fmt/set.h"
#include "utils/containers/require_only_key.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

static DynamicValueAttrs
    mk_value_attrs(size_t src_layer_guid,
                   TensorSlotName src_slot,
                   std::optional<ParallelTensorMapping> const &mapping) {
  return DynamicValueAttrs{
      /*tensor_guid=*/dynamic_tensor_guid_t{
          parallel_tensor_guid_t{
              KwargDataflowOutput{
                  Node{
                      src_layer_guid,
                  },
                  src_slot,
              },
          },
      },
      /*parallel_tensor_shape=*/std::nullopt,
      /*create_grad=*/false,
      /*subgradient_id=*/std::nullopt,
      /*shard_coord=*/std::nullopt,
      /*mapping=*/mapping,
      /*accessor=*/std::nullopt,
      /*role=*/std::nullopt,
  };
}

static DynamicTensorSlot mk_slot(TensorSlotName slot_name) {
  return DynamicTensorSlot{
      /*slot_name=*/slot_name,
      /*slot_tensor_role=*/std::nullopt,
      /*task_shard=*/std::nullopt,
  };
}

static DynamicNodeAttrs mk_node_attrs(size_t layer_guid,
                                      PCGOperatorAttrs const &op_attrs,
                                      DynamicNodeMapping const &mapping) {
  return DynamicNodeAttrs{
      /*task_type=*/std::nullopt,
      /*device_ids=*/std::nullopt,
      /*mapping=*/mapping,
      /*op_attrs=*/
      TrainingOperationAttrs{
          op_attrs,
      },
      /*layer_guid=*/
      dynamic_layer_guid_t{
          parallel_layer_guid_t{
              Node{layer_guid},
          },
      },
      /*per_device_op_state=*/std::nullopt,
  };
}

static MachineSpaceCoordinate mk_machine_coord(nonnegative_int device_idx) {
  return MachineSpaceCoordinate{
      /*node_idx=*/0_n,
      /*device_idx=*/device_idx,
  };
};

static global_device_id_t mk_device_id(MachineSpaceCoordinate const &mc) {
  return global_device_id_t{mc, DeviceType::GPU};
};

static DynamicOpenDataflowGraph
    mk_single_input_node_graph(MachineSpaceCoordinate const &input_device) {
  TensorShape input_shape = TensorShape{
      TensorDims{
          FFOrdered<positive_int>{
              8_p,
              5_p,
          },
      },
      DataType::FLOAT,
  };

  PCGOperatorAttrs input_attrs = PCGOperatorAttrs{
      InputAttrs{
          input_shape,
      },
  };

  PCGOperatorAttrs relu_attrs = PCGOperatorAttrs{
      make_relu_attrs(),
  };

  auto mk_node_mapping =
      [](MappedOperatorTaskGroup const &op_task_group) -> DynamicNodeMapping {
    return DynamicNodeMapping{
        /*op_task_group=*/op_task_group,
        /*device_type=*/DeviceType::GPU,
    };
  };

  auto mk_pt_coord = [](nonnegative_int idx) -> ParallelTensorSpaceCoordinate {
    return ParallelTensorSpaceCoordinate{
        /*sum_component=*/0_n,
        /*discard_copy_component=*/idx,
        /*shared_components=*/
        FFOrdered<nonnegative_int>{
            0_n,
            0_n,
        },
    };
  };

  DynamicValueAttrs input_op_output =
      mk_value_attrs(123, TensorSlotName::OUTPUT, /*mapping=*/std::nullopt);

  MappedOperatorTaskGroup input_node_mapping = MappedOperatorTaskGroup{
      bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding>{
          {
              input_device,
              OperatorAtomicTaskShardBinding{
                  std::map<TensorSlotName, ParallelTensorSpaceCoordinate>{
                      {
                          TensorSlotName::OUTPUT,
                          mk_pt_coord(0_n),
                      },
                  },
              },
          },
      },
  };

  DynamicNodeInvocation input_invocation = DynamicNodeInvocation{
      /*inputs=*/{},
      /*node_attrs=*/
      mk_node_attrs(
          /*layer_guid=*/123,
          /*op_attrs=*/PCGOperatorAttrs{InputAttrs{input_shape}},
          /*mapping=*/mk_node_mapping(input_node_mapping)),
      /*outputs=*/
      {
          {
              mk_slot(TensorSlotName::OUTPUT),
              input_op_output,
          },
      },
  };

  DynamicOpenDataflowGraph g =
      dynamic_open_dataflow_graph_from_invocation_set({input_invocation});

  return g;
}

static DynamicOpenDataflowGraph mk_single_input_into_relu_graph(
    MachineSpaceCoordinate const &input_device,
    MachineSpaceCoordinate const &relu1_device) {
  TensorShape input_shape = TensorShape{
      TensorDims{
          FFOrdered<positive_int>{
              8_p,
              5_p,
          },
      },
      DataType::FLOAT,
  };

  PCGOperatorAttrs input_attrs = PCGOperatorAttrs{
      InputAttrs{
          input_shape,
      },
  };

  PCGOperatorAttrs relu_attrs = PCGOperatorAttrs{
      make_relu_attrs(),
  };

  auto mk_node_mapping =
      [](MappedOperatorTaskGroup const &op_task_group) -> DynamicNodeMapping {
    return DynamicNodeMapping{
        /*op_task_group=*/op_task_group,
        /*device_type=*/DeviceType::GPU,
    };
  };

  auto mk_pt_coord = [](nonnegative_int idx) -> ParallelTensorSpaceCoordinate {
    return ParallelTensorSpaceCoordinate{
        /*sum_component=*/0_n,
        /*discard_copy_component=*/idx,
        /*shared_components=*/
        FFOrdered<nonnegative_int>{
            0_n,
            0_n,
        },
    };
  };

  DynamicValueAttrs input_op_output =
      mk_value_attrs(123, TensorSlotName::OUTPUT, /*mapping=*/std::nullopt);

  MappedOperatorTaskGroup input_node_mapping = MappedOperatorTaskGroup{
      bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding>{
          {
              input_device,
              OperatorAtomicTaskShardBinding{
                  std::map<TensorSlotName, ParallelTensorSpaceCoordinate>{
                      {
                          TensorSlotName::OUTPUT,
                          mk_pt_coord(0_n),
                      },
                  },
              },
          },
      },
  };

  DynamicNodeInvocation input_invocation = DynamicNodeInvocation{
      /*inputs=*/{},
      /*node_attrs=*/
      mk_node_attrs(
          /*layer_guid=*/123,
          /*op_attrs=*/PCGOperatorAttrs{InputAttrs{input_shape}},
          /*mapping=*/mk_node_mapping(input_node_mapping)),
      /*outputs=*/
      {
          {
              mk_slot(TensorSlotName::OUTPUT),
              input_op_output,
          },
      },
  };

  DynamicValueAttrs relu1_op_output =
      mk_value_attrs(124, TensorSlotName::OUTPUT, /*mapping=*/std::nullopt);

  MappedOperatorTaskGroup relu1_node_mapping = MappedOperatorTaskGroup{
      bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding>{
          {
              relu1_device,
              OperatorAtomicTaskShardBinding{
                  std::map<TensorSlotName, ParallelTensorSpaceCoordinate>{
                      {
                          TensorSlotName::INPUT,
                          mk_pt_coord(0_n),
                      },
                      {
                          TensorSlotName::OUTPUT,
                          mk_pt_coord(0_n),
                      },
                  },
              },
          },
      },
  };

  DynamicNodeInvocation relu1_invocation = DynamicNodeInvocation{
      /*inputs=*/{
          {
              mk_slot(TensorSlotName::INPUT),
              input_op_output,
          },
      },
      /*node_attrs=*/
      mk_node_attrs(
          /*layer_guid=*/124,
          /*op_attrs=*/relu_attrs,
          /*mapping=*/mk_node_mapping(relu1_node_mapping)),
      /*outputs=*/
      {
          {
              mk_slot(TensorSlotName::OUTPUT),
              relu1_op_output,
          },
      },
  };

  DynamicOpenDataflowGraph g = dynamic_open_dataflow_graph_from_invocation_set(
      {input_invocation, relu1_invocation});

  return g;
}

struct ExampleGraphTestCase {
  DynamicOpenDataflowGraph g;
  dynamic_invocation_id_t input_op_id;
  dynamic_invocation_id_t relu1_op_id;
  dynamic_invocation_id_t replicate_op_id;
  dynamic_invocation_id_t relu2_op_id;
};

static ExampleGraphTestCase
    mk_example_replicate_graph(MachineSpaceCoordinate const &input_device,
                               MachineSpaceCoordinate const &relu1_device,
                               MachineSpaceCoordinate const &replicate_device1,
                               MachineSpaceCoordinate const &replicate_device2,
                               MachineSpaceCoordinate const &relu2_device1,
                               MachineSpaceCoordinate const &relu2_device2) {
  TensorShape input_shape = TensorShape{
      TensorDims{
          FFOrdered<positive_int>{
              8_p,
              5_p,
          },
      },
      DataType::FLOAT,
  };

  PCGOperatorAttrs input_attrs = PCGOperatorAttrs{
      InputAttrs{
          input_shape,
      },
  };

  PCGOperatorAttrs relu_attrs = PCGOperatorAttrs{
      make_relu_attrs(),
  };

  auto mk_node_mapping =
      [](MappedOperatorTaskGroup const &op_task_group) -> DynamicNodeMapping {
    return DynamicNodeMapping{
        /*op_task_group=*/op_task_group,
        /*device_type=*/DeviceType::GPU,
    };
  };

  auto mk_pt_coord = [](nonnegative_int idx) -> ParallelTensorSpaceCoordinate {
    return ParallelTensorSpaceCoordinate{
        /*sum_component=*/0_n,
        /*discard_copy_component=*/idx,
        /*shared_components=*/
        FFOrdered<nonnegative_int>{
            0_n,
            0_n,
        },
    };
  };

  DynamicValueAttrs input_op_output =
      mk_value_attrs(123, TensorSlotName::OUTPUT, /*mapping=*/std::nullopt);

  MappedOperatorTaskGroup input_node_mapping = MappedOperatorTaskGroup{
      bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding>{
          {
              input_device,
              OperatorAtomicTaskShardBinding{
                  std::map<TensorSlotName, ParallelTensorSpaceCoordinate>{
                      {
                          TensorSlotName::OUTPUT,
                          mk_pt_coord(0_n),
                      },
                  },
              },
          },
      },
  };

  DynamicNodeInvocation input_invocation = DynamicNodeInvocation{
      /*inputs=*/{},
      /*node_attrs=*/
      mk_node_attrs(
          /*layer_guid=*/123,
          /*op_attrs=*/PCGOperatorAttrs{InputAttrs{input_shape}},
          /*mapping=*/mk_node_mapping(input_node_mapping)),
      /*outputs=*/
      {
          {
              mk_slot(TensorSlotName::OUTPUT),
              input_op_output,
          },
      },
  };

  DynamicValueAttrs relu1_op_output =
      mk_value_attrs(124, TensorSlotName::OUTPUT, /*mapping=*/std::nullopt);

  MappedOperatorTaskGroup relu1_node_mapping = MappedOperatorTaskGroup{
      bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding>{
          {
              relu1_device,
              OperatorAtomicTaskShardBinding{
                  std::map<TensorSlotName, ParallelTensorSpaceCoordinate>{
                      {
                          TensorSlotName::INPUT,
                          mk_pt_coord(0_n),
                      },
                      {
                          TensorSlotName::OUTPUT,
                          mk_pt_coord(0_n),
                      },
                  },
              },
          },
      },
  };

  DynamicNodeInvocation relu1_invocation = DynamicNodeInvocation{
      /*inputs=*/{
          {
              mk_slot(TensorSlotName::INPUT),
              input_op_output,
          },
      },
      /*node_attrs=*/
      mk_node_attrs(
          /*layer_guid=*/124,
          /*op_attrs=*/relu_attrs,
          /*mapping=*/mk_node_mapping(relu1_node_mapping)),
      /*outputs=*/
      {
          {
              mk_slot(TensorSlotName::OUTPUT),
              relu1_op_output,
          },
      },
  };

  DynamicValueAttrs replicate_op_output =
      mk_value_attrs(125, TensorSlotName::OUTPUT, /*mapping=*/std::nullopt);

  MappedOperatorTaskGroup replicate_node_mapping = MappedOperatorTaskGroup{
      bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding>{
          {
              replicate_device1,
              OperatorAtomicTaskShardBinding{
                  std::map<TensorSlotName, ParallelTensorSpaceCoordinate>{
                      {
                          TensorSlotName::INPUT,
                          mk_pt_coord(0_n),
                      },
                      {
                          TensorSlotName::OUTPUT,
                          mk_pt_coord(0_n),
                      },
                  },
              },
          },
          {
              replicate_device2,
              OperatorAtomicTaskShardBinding{
                  std::map<TensorSlotName, ParallelTensorSpaceCoordinate>{
                      {
                          TensorSlotName::INPUT,
                          mk_pt_coord(0_n),
                      },
                      {
                          TensorSlotName::OUTPUT,
                          mk_pt_coord(1_n),
                      },
                  },
              },
          },
      },
  };

  DynamicNodeInvocation replicate_invocation = DynamicNodeInvocation{
      /*inputs=*/{
          {
              mk_slot(TensorSlotName::INPUT),
              relu1_op_output,
          },
      },
      /*node_attrs=*/
      mk_node_attrs(
          /*layer_guid=*/125,
          /*op_attrs=*/
          PCGOperatorAttrs{
              ReplicateAttrs{
                  /*replicate_degree=*/2_p,
              },
          },
          /*mapping=*/mk_node_mapping(replicate_node_mapping)),
      /*outputs=*/
      {
          {
              mk_slot(TensorSlotName::OUTPUT),
              replicate_op_output,
          },
      },
  };

  DynamicValueAttrs relu2_op_output =
      mk_value_attrs(126, TensorSlotName::OUTPUT, /*mapping=*/std::nullopt);

  MappedOperatorTaskGroup relu2_node_mapping = MappedOperatorTaskGroup{
      bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding>{
          {
              relu2_device1,
              OperatorAtomicTaskShardBinding{
                  std::map<TensorSlotName, ParallelTensorSpaceCoordinate>{
                      {
                          TensorSlotName::INPUT,
                          mk_pt_coord(0_n),
                      },
                      {
                          TensorSlotName::OUTPUT,
                          mk_pt_coord(0_n),
                      },
                  },
              },
          },
          {
              relu2_device2,
              OperatorAtomicTaskShardBinding{
                  std::map<TensorSlotName, ParallelTensorSpaceCoordinate>{
                      {
                          TensorSlotName::INPUT,
                          mk_pt_coord(1_n),
                      },
                      {
                          TensorSlotName::OUTPUT,
                          mk_pt_coord(1_n),
                      },
                  },
              },
          },
      },
  };

  DynamicNodeInvocation relu2_invocation = DynamicNodeInvocation{
      /*inputs=*/{
          {
              mk_slot(TensorSlotName::INPUT),
              replicate_op_output,
          },
      },
      /*node_attrs=*/
      mk_node_attrs(
          /*layer_guid=*/125,
          /*op_attrs=*/relu_attrs,
          /*mapping=*/mk_node_mapping(relu2_node_mapping)),
      /*outputs=*/
      {
          {
              mk_slot(TensorSlotName::OUTPUT),
              relu2_op_output,
          },
      },
  };

  DynamicOpenDataflowGraph g =
      dynamic_open_dataflow_graph_from_invocation_set({input_invocation,
                                                       relu1_invocation,
                                                       replicate_invocation,
                                                       relu2_invocation});

  return ExampleGraphTestCase{
      /*g=*/g,
      /*input_op_id=*/dynamic_graph_get_id_for_invocation(g, input_invocation),
      /*relu1_op_id=*/dynamic_graph_get_id_for_invocation(g, relu1_invocation),
      /*replicate_op_id=*/
      dynamic_graph_get_id_for_invocation(g, replicate_invocation),
      /*relu2_op_id=*/dynamic_graph_get_id_for_invocation(g, relu2_invocation),
  };
};

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("resolve_tensor_mappings") {
    MachineSpaceCoordinate mc1 = mk_machine_coord(0_n);
    MachineSpaceCoordinate mc2 = mk_machine_coord(1_n);
    MachineSpaceCoordinate mc3 = mk_machine_coord(2_n);

    auto mk_pt_coord =
        [](nonnegative_int idx) -> ParallelTensorSpaceCoordinate {
      return ParallelTensorSpaceCoordinate{
          /*sum_component=*/0_n,
          /*discard_copy_component=*/idx,
          /*shared_components=*/
          FFOrdered<nonnegative_int>{
              0_n,
              0_n,
          },
      };
    };

    SUBCASE("dynamic graph is not pass expanded") {
      auto mk_correct_mappings =
          [&](ExampleGraphTestCase const &tc,
              MachineSpaceCoordinate const &input_out_mc,
              MachineSpaceCoordinate const &relu1_in_mc,
              MachineSpaceCoordinate const &relu1_out_mc,
              MachineSpaceCoordinate const &replicate_in_mc,
              MachineSpaceCoordinate const &replicate_out_mc1,
              MachineSpaceCoordinate const &replicate_out_mc2,
              MachineSpaceCoordinate const &relu2_in_mc1,
              MachineSpaceCoordinate const &relu2_in_mc2,
              MachineSpaceCoordinate const &relu2_out_mc1,
              MachineSpaceCoordinate const &relu2_out_mc2)
          -> std::map<InternalDynamicSlotSite, ParallelTensorMapping> {
        DynamicOpenDataflowGraph g = tc.g;
        dynamic_invocation_id_t input_op_id = tc.input_op_id;
        dynamic_invocation_id_t relu1_op_id = tc.relu1_op_id;
        dynamic_invocation_id_t replicate_op_id = tc.replicate_op_id;
        dynamic_invocation_id_t relu2_op_id = tc.relu2_op_id;

        auto mk_inp_slot = [](dynamic_invocation_id_t invocation_id)
            -> InternalDynamicSlotSite {
          return InternalDynamicSlotSite{
              /*invocation_id=*/invocation_id,
              /*direction=*/TensorDirection::INCOMING,
              /*slot_name=*/mk_slot(TensorSlotName::INPUT),
          };
        };

        auto mk_out_slot = [](dynamic_invocation_id_t invocation_id)
            -> InternalDynamicSlotSite {
          return InternalDynamicSlotSite{
              /*invocation_id=*/invocation_id,
              /*direction=*/TensorDirection::OUTPUT,
              /*slot_name=*/mk_slot(TensorSlotName::OUTPUT),
          };
        };

        auto mk_single_shard_mapping =
            [&](MachineSpaceCoordinate const &mc) -> ParallelTensorMapping {
          return ParallelTensorMapping{
              bidict<ParallelTensorSpaceCoordinate, global_device_id_t>{
                  {
                      mk_pt_coord(0_n),
                      mk_device_id(mc),
                  },
              },
          };
        };

        auto mk_two_shard_mapping =
            [&](MachineSpaceCoordinate const &mc1,
                MachineSpaceCoordinate const &mc2) -> ParallelTensorMapping {
          return ParallelTensorMapping{
              bidict<ParallelTensorSpaceCoordinate, global_device_id_t>{
                  {
                      mk_pt_coord(0_n),
                      mk_device_id(mc1),
                  },
                  {
                      mk_pt_coord(1_n),
                      mk_device_id(mc2),
                  },
              },
          };
        };

        InternalDynamicSlotSite input_op_out = mk_out_slot(input_op_id);
        InternalDynamicSlotSite relu1_op_in = mk_inp_slot(relu1_op_id);
        InternalDynamicSlotSite relu1_op_out = mk_out_slot(relu1_op_id);
        InternalDynamicSlotSite replicate_op_in = mk_inp_slot(replicate_op_id);
        InternalDynamicSlotSite replicate_op_out = mk_out_slot(replicate_op_id);
        InternalDynamicSlotSite relu2_op_in = mk_inp_slot(relu2_op_id);
        InternalDynamicSlotSite relu2_op_out = mk_out_slot(relu2_op_id);

        return {
            {
                input_op_out,
                mk_single_shard_mapping(input_out_mc),
            },
            {
                relu1_op_in,
                mk_single_shard_mapping(relu1_in_mc),
            },
            {
                relu1_op_out,
                mk_single_shard_mapping(relu1_out_mc),
            },
            {
                replicate_op_in,
                mk_single_shard_mapping(replicate_in_mc),
            },
            {
                replicate_op_out,
                mk_two_shard_mapping(replicate_out_mc1, replicate_out_mc2),
            },
            {
                relu2_op_in,
                mk_two_shard_mapping(relu2_in_mc1, relu2_in_mc2),
            },
            {
                relu2_op_out,
                mk_two_shard_mapping(relu2_out_mc1, relu2_out_mc2),
            },
        };
      };

      SUBCASE("replicate input matches tensor source") {
        ExampleGraphTestCase tc = mk_example_replicate_graph(
            /*input_device=*/mc1,
            /*relu1_device=*/mc1,
            /*replicate_device1=*/mc1,
            /*replicate_device2=*/mc2,
            /*relu2_device1=*/mc1,
            /*relu2_device2=*/mc2);

        std::map<InternalDynamicSlotSite, ParallelTensorMapping> result =
            resolve_tensor_mappings(tc.g);

        std::map<InternalDynamicSlotSite, ParallelTensorMapping> correct =
            mk_correct_mappings(
                /*tc=*/tc,
                /*input_out_mc=*/mc1,
                /*relu1_in_mc=*/mc1,
                /*relu1_out_mc=*/mc1,
                /*replicate_in_mc=*/mc1,
                /*relicate_out_mc1=*/mc1,
                /*relicate_out_mc2=*/mc2,
                /*relu2_in_mc1=*/mc1,
                /*relu2_in_mc2=*/mc2,
                /*relu2_out_mc1=*/mc1,
                /*relu2_out_mc1=*/mc2);

        ASSERT(result == correct);
      }

      SUBCASE("input invocation's output follows the input's mapping") {
        ExampleGraphTestCase tc = mk_example_replicate_graph(
            /*input_device=*/mc3,
            /*relu1_device=*/mc1,
            /*replicate_device1=*/mc1,
            /*replicate_device2=*/mc2,
            /*relu2_device1=*/mc1,
            /*relu2_device2=*/mc2);

        std::map<InternalDynamicSlotSite, ParallelTensorMapping> result =
            resolve_tensor_mappings(tc.g);

        std::map<InternalDynamicSlotSite, ParallelTensorMapping> correct =
            mk_correct_mappings(
                /*tc=*/tc,
                /*input_out_mc=*/mc3,
                /*relu1_in_mc=*/mc1,
                /*relu1_out_mc=*/mc1,
                /*replicate_in_mc=*/mc1,
                /*relicate_out_mc1=*/mc1,
                /*relicate_out_mc2=*/mc2,
                /*relu2_in_mc1=*/mc1,
                /*relu2_in_mc2=*/mc2,
                /*relu2_out_mc1=*/mc1,
                /*relu2_out_mc1=*/mc2);

        ASSERT(result == correct);
      }

      SUBCASE("src and sink can differ due to different invocation mappings") {
        ExampleGraphTestCase tc = mk_example_replicate_graph(
            /*input_device=*/mc3,
            /*relu1_device=*/mc1,
            /*replicate_device1=*/mc2,
            /*replicate_device2=*/mc3,
            /*relu2_device1=*/mc1,
            /*relu2_device2=*/mc3);

        std::map<InternalDynamicSlotSite, ParallelTensorMapping> result =
            resolve_tensor_mappings(tc.g);

        std::map<InternalDynamicSlotSite, ParallelTensorMapping> correct =
            mk_correct_mappings(
                /*tc=*/tc,
                /*input_out_mc=*/mc3,
                /*relu1_in_mc=*/mc1,
                /*relu1_out_mc=*/mc1,
                /*replicate_in_mc=*/mc1,
                /*relicate_out_mc1=*/mc2,
                /*relicate_out_mc2=*/mc3,
                /*relu2_in_mc1=*/mc1,
                /*relu2_in_mc2=*/mc3,
                /*relu2_out_mc1=*/mc1,
                /*relu2_out_mc2=*/mc3);

        ASSERT(result == correct);
      }
    }
  }

  TEST_CASE("copies_for_value") {
    DynamicValueAttrs value_attrs = DynamicValueAttrs{
        /*tensor_guid=*/dynamic_tensor_guid_t{
            parallel_tensor_guid_t{
                KwargDataflowOutput<TensorSlotName>{
                    Node{1},
                    TensorSlotName::OUTPUT,
                },
            },
        },
        /*parallel_tensor_shape=*/std::nullopt,
        /*create_grad=*/std::nullopt,
        /*subgradient_id=*/std::nullopt,
        /*shard_coord=*/std::nullopt,
        /*mapping=*/std::nullopt,
        /*accessor=*/std::nullopt,
        /*role=*/std::nullopt,
    };

    auto mk_pt_coord =
        [](nonnegative_int idx) -> ParallelTensorSpaceCoordinate {
      return ParallelTensorSpaceCoordinate{
          /*sum_component=*/0_n,
          /*discard_copy_component=*/0_n,
          /*shared_components=*/
          FFOrdered<nonnegative_int>{
              idx,
              0_n,
          },
      };
    };

    auto mk_device = [](nonnegative_int idx) -> global_device_id_t {
      return global_device_id_t{
          /*coord=*/MachineSpaceCoordinate{
              /*node_idx=*/2_n,
              /*device_idx=*/idx,
          },
          /*device_type=*/DeviceType::GPU,
      };
    };

    auto mk_slot_site = [](nonnegative_int idx) -> InternalDynamicSlotSite {
      return InternalDynamicSlotSite{
          /*invocation_id=*/dynamic_invocation_id_t{idx},
          /*direction=*/TensorDirection::INCOMING,
          /*slot_name=*/
          DynamicTensorSlot{
              /*slot_name=*/TensorSlotName::INPUT,
              /*slot_tensor_role=*/
              DynamicTensorRole{FwbTensorType::FORWARD}, // could be any role
              /*task_shard=*/std::nullopt,
          },
      };
    };

    SUBCASE("if src site is external, no copies no matter what") {
      DynamicSlotSite src_site = DynamicSlotSite{
          ExternalDynamicSlotSite{
              dynamic_external_value_id_t{0_n},
          },
      };

      ParallelTensorMapping mapping1 = ParallelTensorMapping{
          bidict<ParallelTensorSpaceCoordinate, global_device_id_t>{
              {mk_pt_coord(0_n), mk_device(0_n)},
          },
      };

      ParallelTensorMapping mapping2 = ParallelTensorMapping{
          bidict<ParallelTensorSpaceCoordinate, global_device_id_t>{
              {mk_pt_coord(0_n), mk_device(1_n)},
          },
      };

      ParallelTensorMapping mapping3 = ParallelTensorMapping{
          bidict<ParallelTensorSpaceCoordinate, global_device_id_t>{
              {mk_pt_coord(0_n), mk_device(2_n)},
          },
      };

      InternalDynamicSlotSite dst_site1 = mk_slot_site(2_n);
      InternalDynamicSlotSite dst_site2 = mk_slot_site(3_n);
      InternalDynamicSlotSite dst_site3 = mk_slot_site(4_n);

      std::map<InternalDynamicSlotSite, ParallelTensorMapping> site_mappings = {
          {dst_site1, mapping2},
          {dst_site2, mapping1},
          {dst_site3, mapping3},
      };

      std::set<DynamicValueCopyInfo> result = copies_for_value(
          /*value_attrs=*/value_attrs,
          /*src_site=*/DynamicSlotSite{src_site},
          /*dst_sites=*/{dst_site1, dst_site2, dst_site3},
          /*all_mappings=*/site_mappings);

      std::set<DynamicValueCopyInfo> correct = {};

      CHECK(result == correct);
    };

    InternalDynamicSlotSite src_site = InternalDynamicSlotSite{
        /*invocation_id=*/dynamic_invocation_id_t{0_n},
        /*direction=*/TensorDirection::OUTPUT,
        /*slot_name=*/
        DynamicTensorSlot{
            /*slot_name=*/TensorSlotName::OUTPUT,
            /*slot_tensor_role=*/std::nullopt,
            /*task_shard=*/std::nullopt,
        },
    };

    SUBCASE("if src mapping is same as dst mapping don't copy") {
      ParallelTensorMapping mapping1 = ParallelTensorMapping{
          bidict<ParallelTensorSpaceCoordinate, global_device_id_t>{
              {mk_pt_coord(0_n), mk_device(0_n)},
          },
      };

      InternalDynamicSlotSite dst_site = mk_slot_site(1_n);

      std::map<InternalDynamicSlotSite, ParallelTensorMapping> site_mappings = {
          {src_site, mapping1},
          {dst_site, mapping1},
      };

      std::set<DynamicValueCopyInfo> result = copies_for_value(
          /*value_attrs=*/value_attrs,
          /*src_site=*/DynamicSlotSite{src_site},
          /*dst_sites=*/{dst_site},
          /*all_mappings=*/site_mappings);

      std::set<DynamicValueCopyInfo> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("if src mapping does not overlap dst mapping issue copy") {
      ParallelTensorMapping mapping1 = ParallelTensorMapping{
          bidict<ParallelTensorSpaceCoordinate, global_device_id_t>{
              {mk_pt_coord(0_n), mk_device(0_n)},
          },
      };

      ParallelTensorMapping mapping2 = ParallelTensorMapping{
          bidict<ParallelTensorSpaceCoordinate, global_device_id_t>{
              {mk_pt_coord(0_n), mk_device(5_n)},
          },
      };

      InternalDynamicSlotSite dst_site = mk_slot_site(1_n);

      std::map<InternalDynamicSlotSite, ParallelTensorMapping> site_mappings = {
          {src_site, mapping1},
          {dst_site, mapping2},
      };

      std::set<DynamicValueCopyInfo> result = copies_for_value(
          /*value_attrs=*/value_attrs,
          /*src_site=*/DynamicSlotSite{src_site},
          /*dst_sites=*/{dst_site},
          /*all_mappings=*/site_mappings);

      std::set<DynamicValueCopyInfo> correct = {
          DynamicValueCopyInfo{
              /*value_attrs=*/value_attrs,
              /*src_mapping=*/mapping1,
              /*dst_mapping=*/mapping2,
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("if src mapping overlaps dst mapping issue full copy") {
      ParallelTensorMapping mapping1 = ParallelTensorMapping{
          bidict<ParallelTensorSpaceCoordinate, global_device_id_t>{
              {mk_pt_coord(0_n), mk_device(0_n)},
              {mk_pt_coord(1_n), mk_device(1_n)},
          },
      };

      ParallelTensorMapping mapping2 = ParallelTensorMapping{
          bidict<ParallelTensorSpaceCoordinate, global_device_id_t>{
              {mk_pt_coord(0_n), mk_device(0_n)},
              {mk_pt_coord(1_n), mk_device(2_n)},
          },
      };

      InternalDynamicSlotSite dst_site = mk_slot_site(1_n);

      std::map<InternalDynamicSlotSite, ParallelTensorMapping> site_mappings = {
          {src_site, mapping1},
          {dst_site, mapping2},
      };

      std::set<DynamicValueCopyInfo> result = copies_for_value(
          /*value_attrs=*/value_attrs,
          /*src_site=*/DynamicSlotSite{src_site},
          /*dst_sites=*/{dst_site},
          /*all_mappings=*/site_mappings);

      std::set<DynamicValueCopyInfo> correct = {
          DynamicValueCopyInfo{
              /*value_attrs=*/value_attrs,
              /*src_mapping=*/mapping1,
              /*dst_mapping=*/mapping2,
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("if src mapping overlaps multiple dst mappings issue both copies") {
      ParallelTensorMapping mapping1 = ParallelTensorMapping{
          bidict<ParallelTensorSpaceCoordinate, global_device_id_t>{
              {mk_pt_coord(0_n), mk_device(0_n)},
              {mk_pt_coord(1_n), mk_device(1_n)},
          },
      };

      ParallelTensorMapping mapping2 = ParallelTensorMapping{
          bidict<ParallelTensorSpaceCoordinate, global_device_id_t>{
              {mk_pt_coord(0_n), mk_device(0_n)},
              {mk_pt_coord(1_n), mk_device(2_n)},
          },
      };

      ParallelTensorMapping mapping3 = ParallelTensorMapping{
          bidict<ParallelTensorSpaceCoordinate, global_device_id_t>{
              {mk_pt_coord(0_n), mk_device(0_n)},
              {mk_pt_coord(1_n), mk_device(3_n)},
          },
      };

      InternalDynamicSlotSite dst_site1 = mk_slot_site(1_n);
      InternalDynamicSlotSite dst_site2 = mk_slot_site(2_n);

      std::map<InternalDynamicSlotSite, ParallelTensorMapping> site_mappings = {
          {src_site, mapping1},
          {dst_site1, mapping2},
          {dst_site2, mapping3},
      };

      std::set<DynamicValueCopyInfo> result = copies_for_value(
          /*value_attrs=*/value_attrs,
          /*src_site=*/DynamicSlotSite{src_site},
          /*dst_sites=*/{dst_site1, dst_site2},
          /*sink_site_mappings=*/site_mappings);

      std::set<DynamicValueCopyInfo> correct = {
          DynamicValueCopyInfo{
              /*value_attrs=*/value_attrs,
              /*src_mapping=*/mapping1,
              /*dst_mapping=*/mapping2,
          },
          DynamicValueCopyInfo{
              /*value_attrs=*/value_attrs,
              /*src_mapping=*/mapping1,
              /*dst_mapping=*/mapping3,
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("only copy once if multiple sinks use the same mapping") {
      ParallelTensorMapping mapping1 = ParallelTensorMapping{
          bidict<ParallelTensorSpaceCoordinate, global_device_id_t>{
              {mk_pt_coord(0_n), mk_device(0_n)},
          },
      };

      ParallelTensorMapping mapping2 = ParallelTensorMapping{
          bidict<ParallelTensorSpaceCoordinate, global_device_id_t>{
              {mk_pt_coord(0_n), mk_device(5_n)},
          },
      };

      InternalDynamicSlotSite dst_site1 = mk_slot_site(1_n);
      InternalDynamicSlotSite dst_site2 = mk_slot_site(2_n);
      InternalDynamicSlotSite dst_site3 = mk_slot_site(3_n);

      std::map<InternalDynamicSlotSite, ParallelTensorMapping> site_mappings = {
          {src_site, mapping1},
          {dst_site1, mapping2},
          {dst_site2, mapping2},
          {dst_site3, mapping2},
      };

      std::set<DynamicValueCopyInfo> result = copies_for_value(
          /*value_attrs=*/value_attrs,
          /*src_site=*/DynamicSlotSite{src_site},
          /*dst_sites=*/{dst_site1, dst_site2, dst_site3},
          /*all_mappings=*/site_mappings);

      std::set<DynamicValueCopyInfo> correct = {
          DynamicValueCopyInfo{
              /*value_attrs=*/value_attrs,
              /*src_mapping=*/mapping1,
              /*dst_mapping=*/mapping2,
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("if src mapping matches one dst mapping, still issue copies for "
            "the rest") {
      ParallelTensorMapping mapping1 = ParallelTensorMapping{
          bidict<ParallelTensorSpaceCoordinate, global_device_id_t>{
              {mk_pt_coord(0_n), mk_device(0_n)},
          },
      };

      ParallelTensorMapping mapping2 = ParallelTensorMapping{
          bidict<ParallelTensorSpaceCoordinate, global_device_id_t>{
              {mk_pt_coord(0_n), mk_device(5_n)},
          },
      };

      ParallelTensorMapping mapping3 = ParallelTensorMapping{
          bidict<ParallelTensorSpaceCoordinate, global_device_id_t>{
              {mk_pt_coord(0_n), mk_device(6_n)},
          },
      };

      ParallelTensorMapping mapping4 = ParallelTensorMapping{
          bidict<ParallelTensorSpaceCoordinate, global_device_id_t>{
              {mk_pt_coord(0_n), mk_device(7_n)},
          },
      };

      InternalDynamicSlotSite dst_site1 = mk_slot_site(1_n);
      InternalDynamicSlotSite dst_site2 = mk_slot_site(2_n);
      InternalDynamicSlotSite dst_site3 = mk_slot_site(3_n);
      InternalDynamicSlotSite dst_site4 = mk_slot_site(4_n);
      InternalDynamicSlotSite dst_site5 = mk_slot_site(5_n);

      std::map<InternalDynamicSlotSite, ParallelTensorMapping> site_mappings = {
          {src_site, mapping1},
          {dst_site1, mapping2},
          {dst_site2, mapping1},
          {dst_site3, mapping2},
          {dst_site4, mapping4},
          {dst_site5, mapping3},
      };

      std::set<DynamicValueCopyInfo> result = copies_for_value(
          /*value_attrs=*/value_attrs,
          /*src_site=*/DynamicSlotSite{src_site},
          /*dst_sites=*/{dst_site1, dst_site2, dst_site3, dst_site4, dst_site5},
          /*all_mappings=*/site_mappings);

      std::set<DynamicValueCopyInfo> correct = {
          DynamicValueCopyInfo{
              /*value_attrs=*/value_attrs,
              /*src_mapping=*/mapping1,
              /*dst_mapping=*/mapping2,
          },
          DynamicValueCopyInfo{
              /*value_attrs=*/value_attrs,
              /*src_mapping=*/mapping1,
              /*dst_mapping=*/mapping3,
          },
          DynamicValueCopyInfo{
              /*value_attrs=*/value_attrs,
              /*src_mapping=*/mapping1,
              /*dst_mapping=*/mapping4,
          },
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("perform_copy_insertion") {

    SUBCASE("standard operator") {
      auto mk_ptensor_coord =
          [](nonnegative_int shard_idx) -> ParallelTensorSpaceCoordinate {
        return ParallelTensorSpaceCoordinate{
            /*sum_component=*/0_n,
            /*discard_copy_component=*/0_n,
            /*shard_components=*/
            FFOrdered<nonnegative_int>{
                shard_idx,
            },
        };
      };

      auto mk_device_id =
          [&](nonnegative_int device_idx) -> global_device_id_t {
        return global_device_id_t{
            mk_machine_coord(device_idx),
            DeviceType::GPU,
        };
      };

      auto mk_pcg_layer_guid =
          [](size_t pcg_layer_guid) -> dynamic_layer_guid_t {
        return dynamic_layer_guid_t{
            parallel_layer_guid_t{
                Node{pcg_layer_guid},
            },
        };
      };

      auto mk_node_attrs =
          [](dynamic_layer_guid_t layer_guid,
             std::optional<DynamicNodeMapping> const &mapping,
             TrainingOperationAttrs const &op_attrs) -> DynamicNodeAttrs {
        return DynamicNodeAttrs{
            /*task_type=*/std::nullopt,
            /*device_coord=*/std::nullopt,
            /*mapping=*/mapping,
            /*op_attrs=*/op_attrs,
            /*layer_guid=*/layer_guid,
            /*per_device_op_state=*/std::nullopt,
        };
      };

      auto mk_binding = [&](nonnegative_int input_shard_idx,
                            nonnegative_int output_shard_idx)
          -> OperatorAtomicTaskShardBinding {
        return OperatorAtomicTaskShardBinding{
            /*tensor_coords=*/std::map<TensorSlotName,
                                       ParallelTensorSpaceCoordinate>{
                {
                    TensorSlotName::INPUT,
                    mk_ptensor_coord(input_shard_idx),
                },
                {
                    TensorSlotName::OUTPUT,
                    mk_ptensor_coord(output_shard_idx),
                },
            },
        };
      };

      TrainingOperationAttrs relu_attrs = TrainingOperationAttrs{
          PCGOperatorAttrs{
              make_relu_attrs(),
          },
      };

      DynamicValueAttrs v1 = mk_value_attrs(
          /*src_layer_guid=*/0,
          /*src_slot=*/TensorSlotName::OUTPUT,
          /*mapping=*/std::nullopt);

      DynamicValueAttrs v2 = mk_value_attrs(
          /*src_layer_guid=*/1,
          /*src_slot=*/TensorSlotName::OUTPUT,
          /*mapping=*/std::nullopt);

      DynamicValueAttrs v3 = mk_value_attrs(
          /*src_layer_guid=*/2,
          /*src_slot=*/TensorSlotName::OUTPUT,
          /*mapping=*/std::nullopt);

      SUBCASE("inserts copy when necessary") {
        DynamicNodeMapping mapping1 = DynamicNodeMapping{
            MappedOperatorTaskGroup{
                bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding>{
                    {
                        mk_machine_coord(0_n),
                        mk_binding(0_n, 0_n),
                    },
                    {
                        mk_machine_coord(1_n),
                        mk_binding(1_n, 1_n),
                    },
                },
            },
            DeviceType::GPU,
        };

        DynamicNodeMapping mapping2 = DynamicNodeMapping{
            MappedOperatorTaskGroup{
                bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding>{
                    {
                        mk_machine_coord(0_n),
                        mk_binding(0_n, 0_n),
                    },
                    {
                        mk_machine_coord(2_n),
                        mk_binding(1_n, 1_n),
                    },
                },
            },
            DeviceType::GPU,
        };

        DynamicNodeInvocation inv1 = DynamicNodeInvocation{
            /*inputs=*/{
                {
                    mk_slot(TensorSlotName::INPUT),
                    v1,
                },
            },
            /*node_attrs=*/
            mk_node_attrs(mk_pcg_layer_guid(1), mapping1, relu_attrs),
            /*outputs=*/
            {
                {
                    mk_slot(TensorSlotName::OUTPUT),
                    v2,
                },
            },
        };

        DynamicNodeInvocation inv2 = DynamicNodeInvocation{
            /*inputs=*/{
                {
                    mk_slot(TensorSlotName::INPUT),
                    v2,
                },
            },
            /*node_attrs=*/
            mk_node_attrs(mk_pcg_layer_guid(2), mapping2, relu_attrs),
            /*outputs=*/
            {
                {
                    mk_slot(TensorSlotName::OUTPUT),
                    v3,
                },
            },
        };

        DynamicOpenDataflowGraph g =
            dynamic_open_dataflow_graph_from_invocation_set({inv1, inv2});

        DynamicOpenDataflowGraph result = perform_copy_insertion(g);

        DynamicOpenDataflowGraph correct = [&] {
          DynamicValueAttrs mapped_v1 = mk_value_attrs(
              /*src_layer_guid=*/0,
              /*src_slot=*/TensorSlotName::OUTPUT,
              /*mapping=*/
              ParallelTensorMapping{
                  bidict<ParallelTensorSpaceCoordinate, global_device_id_t>{
                      {mk_ptensor_coord(0_n), mk_device_id(0_n)},
                      {mk_ptensor_coord(1_n), mk_device_id(1_n)},
                  },
              });

          DynamicValueAttrs mapped_v2_placement1 = mk_value_attrs(
              /*src_layer_guid=*/1,
              /*src_slot=*/TensorSlotName::OUTPUT,
              /*mapping=*/
              ParallelTensorMapping{
                  bidict<ParallelTensorSpaceCoordinate, global_device_id_t>{
                      {mk_ptensor_coord(0_n), mk_device_id(0_n)},
                      {mk_ptensor_coord(1_n), mk_device_id(1_n)},
                  },
              });

          DynamicValueAttrs mapped_v2_placement2 = mk_value_attrs(
              /*src_layer_guid=*/1,
              /*src_slot=*/TensorSlotName::OUTPUT,
              /*mapping=*/
              ParallelTensorMapping{
                  bidict<ParallelTensorSpaceCoordinate, global_device_id_t>{
                      {mk_ptensor_coord(0_n), mk_device_id(0_n)},
                      {mk_ptensor_coord(1_n), mk_device_id(2_n)},
                  },
              });

          DynamicValueAttrs mapped_v3 = mk_value_attrs(
              /*src_layer_guid=*/2,
              /*src_slot=*/TensorSlotName::OUTPUT,
              /*mapping=*/
              ParallelTensorMapping{
                  bidict<ParallelTensorSpaceCoordinate, global_device_id_t>{
                      {mk_ptensor_coord(0_n), mk_device_id(0_n)},
                      {mk_ptensor_coord(1_n), mk_device_id(2_n)},
                  },
              });

          DynamicNodeInvocation mapped_inv1 = DynamicNodeInvocation{
              /*inputs=*/{
                  {
                      mk_slot(TensorSlotName::INPUT),
                      mapped_v1,
                  },
              },
              /*node_attrs=*/
              mk_node_attrs(mk_pcg_layer_guid(1), mapping1, relu_attrs),
              /*outputs=*/
              {
                  {
                      mk_slot(TensorSlotName::OUTPUT),
                      mapped_v2_placement1,
                  },
              },
          };

          DynamicNodeInvocation inserted_copy = DynamicNodeInvocation{
              /*inputs=*/{
                  {
                      mk_slot(TensorSlotName::INPUT),
                      mapped_v2_placement1,
                  },
              },
              /*node_attrs=*/
              mk_node_attrs(dynamic_layer_guid_t{dynamic_copy_layer_guid_t{}},
                            std::nullopt,
                            /*op_attrs=*/TrainingOperationAttrs{CopyAttrs{}}),
              /*outputs=*/
              {
                  {
                      mk_slot(TensorSlotName::OUTPUT),
                      mapped_v2_placement2,
                  },
              },

          };

          DynamicNodeInvocation mapped_inv2 = DynamicNodeInvocation{
              /*inputs=*/{
                  {
                      mk_slot(TensorSlotName::INPUT),
                      mapped_v2_placement2,
                  },
              },
              /*node_attrs=*/
              mk_node_attrs(mk_pcg_layer_guid(2), mapping2, relu_attrs),
              /*outputs=*/
              {
                  {
                      mk_slot(TensorSlotName::OUTPUT),
                      mapped_v3,
                  },
              },
          };

          return dynamic_open_dataflow_graph_from_invocation_set(
              {mapped_inv1, mapped_inv2, inserted_copy});
        }();

        CHECK_MESSAGE(
            result == correct,
            check_kv("result\n", dynamic_open_dataflow_graph_as_dot(result)),
            check_kv("correct\n", dynamic_open_dataflow_graph_as_dot(correct)));
      }

      SUBCASE("does not insert a copy when not necessary") {
        DynamicNodeMapping mapping1 = DynamicNodeMapping{
            MappedOperatorTaskGroup{
                bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding>{
                    {
                        mk_machine_coord(0_n),
                        mk_binding(0_n, 0_n),
                    },
                    {
                        mk_machine_coord(1_n),
                        mk_binding(1_n, 1_n),
                    },
                },
            },
            DeviceType::GPU,
        };

        DynamicNodeMapping mapping2 = DynamicNodeMapping{
            MappedOperatorTaskGroup{
                bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding>{
                    {
                        mk_machine_coord(0_n),
                        mk_binding(0_n, 0_n),
                    },
                    {
                        mk_machine_coord(1_n),
                        mk_binding(1_n, 1_n),
                    },
                },
            },
            DeviceType::GPU,
        };

        DynamicNodeInvocation inv1 = DynamicNodeInvocation{
            /*inputs=*/{
                {
                    mk_slot(TensorSlotName::INPUT),
                    v1,
                },
            },
            /*node_attrs=*/
            mk_node_attrs(mk_pcg_layer_guid(1), mapping1, relu_attrs),
            /*outputs=*/
            {
                {
                    mk_slot(TensorSlotName::OUTPUT),
                    v2,
                },
            },
        };

        DynamicNodeInvocation inv2 = DynamicNodeInvocation{
            /*inputs=*/{
                {
                    mk_slot(TensorSlotName::INPUT),
                    v2,
                },
            },
            /*node_attrs=*/
            mk_node_attrs(mk_pcg_layer_guid(2), mapping2, relu_attrs),
            /*outputs=*/
            {
                {
                    mk_slot(TensorSlotName::OUTPUT),
                    v3,
                },
            },
        };

        DynamicOpenDataflowGraph g =
            dynamic_open_dataflow_graph_from_invocation_set({inv1, inv2});

        DynamicOpenDataflowGraph result = perform_copy_insertion(g);

        DynamicOpenDataflowGraph correct = [&] {
          DynamicValueAttrs mapped_v1 = mk_value_attrs(
              /*src_layer_guid=*/0,
              /*src_slot=*/TensorSlotName::OUTPUT,
              /*mapping=*/
              ParallelTensorMapping{
                  bidict<ParallelTensorSpaceCoordinate, global_device_id_t>{
                      {mk_ptensor_coord(0_n), mk_device_id(0_n)},
                      {mk_ptensor_coord(1_n), mk_device_id(1_n)},
                  },
              });

          DynamicValueAttrs mapped_v2 = mk_value_attrs(
              /*src_layer_guid=*/1,
              /*src_slot=*/TensorSlotName::OUTPUT,
              /*mapping=*/
              ParallelTensorMapping{
                  bidict<ParallelTensorSpaceCoordinate, global_device_id_t>{
                      {mk_ptensor_coord(0_n), mk_device_id(0_n)},
                      {mk_ptensor_coord(1_n), mk_device_id(1_n)},
                  },
              });

          DynamicValueAttrs mapped_v3 = mk_value_attrs(
              /*src_layer_guid=*/2,
              /*src_slot=*/TensorSlotName::OUTPUT,
              /*mapping=*/
              ParallelTensorMapping{
                  bidict<ParallelTensorSpaceCoordinate, global_device_id_t>{
                      {mk_ptensor_coord(0_n), mk_device_id(0_n)},
                      {mk_ptensor_coord(1_n), mk_device_id(1_n)},
                  },
              });

          DynamicNodeInvocation mapped_inv1 = DynamicNodeInvocation{
              /*inputs=*/{
                  {
                      mk_slot(TensorSlotName::INPUT),
                      mapped_v1,
                  },
              },
              /*node_attrs=*/
              mk_node_attrs(mk_pcg_layer_guid(1), mapping1, relu_attrs),
              /*outputs=*/
              {
                  {
                      mk_slot(TensorSlotName::OUTPUT),
                      mapped_v2,
                  },
              },
          };

          DynamicNodeInvocation mapped_inv2 = DynamicNodeInvocation{
              /*inputs=*/{
                  {
                      mk_slot(TensorSlotName::INPUT),
                      mapped_v2,
                  },
              },
              /*node_attrs=*/
              mk_node_attrs(mk_pcg_layer_guid(2), mapping2, relu_attrs),
              /*outputs=*/
              {
                  {
                      mk_slot(TensorSlotName::OUTPUT),
                      mapped_v3,
                  },
              },
          };

          return dynamic_open_dataflow_graph_from_invocation_set(
              {mapped_inv1, mapped_inv2});
        }();

        CHECK_MESSAGE(
            result == correct,
            check_kv("result\n", dynamic_open_dataflow_graph_as_dot(result)),
            check_kv("correct\n", dynamic_open_dataflow_graph_as_dot(correct)));
      }
    }

    SUBCASE("replicate operator") {
      auto mk_pt_coord =
          [](nonnegative_int idx) -> ParallelTensorSpaceCoordinate {
        return ParallelTensorSpaceCoordinate{
            /*sum_component=*/0_n,
            /*discard_copy_component=*/idx,
            /*shared_components=*/
            FFOrdered<nonnegative_int>{
                0_n,
                0_n,
            },
        };
      };

      MachineSpaceCoordinate mc1 = mk_machine_coord(0_n);
      MachineSpaceCoordinate mc2 = mk_machine_coord(1_n);
      MachineSpaceCoordinate mc3 = mk_machine_coord(2_n);

      ExampleGraphTestCase tc = mk_example_replicate_graph(
          /*input_device=*/mc3,
          /*relu1_device=*/mc1,
          /*replicate_device1=*/mc2,
          /*replicate_device2=*/mc3,
          /*relu2_device1=*/mc1,
          /*relu2_device2=*/mc3);

      std::set<DynamicValueCopyInfo> copies = infer_all_copies_in_graph(tc.g);
      ASSERT(copies.size() == 2);

      DynamicOpenDataflowGraph result = perform_copy_insertion(tc.g);

      DynamicOpenDataflowGraph correct = [&] {
        auto map_input_value =
            [](DynamicNodeInvocation const &invocation,
               ParallelTensorMapping const &mapping) -> DynamicNodeInvocation {
          DynamicNodeInvocation result = invocation;
          DynamicTensorSlot input_slot = mk_slot(TensorSlotName::INPUT);
          result.inputs = {
              {
                  input_slot,
                  decide_dynamic_value_attrs_mapping(
                      require_only_key(invocation.inputs, input_slot), mapping),
              },
          };
          return result;
        };

        auto map_output_value =
            [](DynamicNodeInvocation const &invocation,
               ParallelTensorMapping const &mapping) -> DynamicNodeInvocation {
          DynamicNodeInvocation result = invocation;
          DynamicTensorSlot output_slot = mk_slot(TensorSlotName::OUTPUT);
          result.outputs = {
              {
                  output_slot,
                  decide_dynamic_value_attrs_mapping(
                      require_only_key(invocation.outputs, output_slot),
                      mapping),
              },
          };
          return result;
        };

        auto map_input_and_output_values =
            [&](DynamicNodeInvocation const &invocation,
                ParallelTensorMapping const &input_mapping,
                ParallelTensorMapping const &output_mapping)
            -> DynamicNodeInvocation {
          return map_input_value(map_output_value(invocation, output_mapping),
                                 input_mapping);
        };

        auto mk_single_shard_mapping =
            [&](MachineSpaceCoordinate const &mc) -> ParallelTensorMapping {
          return ParallelTensorMapping{
              bidict<ParallelTensorSpaceCoordinate, global_device_id_t>{
                  {
                      mk_pt_coord(0_n),
                      mk_device_id(mc),
                  },
              },
          };
        };

        auto mk_two_shard_mapping =
            [&](MachineSpaceCoordinate const &mc1,
                MachineSpaceCoordinate const &mc2) -> ParallelTensorMapping {
          return ParallelTensorMapping{
              bidict<ParallelTensorSpaceCoordinate, global_device_id_t>{
                  {
                      mk_pt_coord(0_n),
                      mk_device_id(mc1),
                  },
                  {
                      mk_pt_coord(1_n),
                      mk_device_id(mc2),
                  },
              },
          };
        };

        DynamicNodeInvocation input_invocation =
            dynamic_graph_get_invocation_for_id(tc.g, tc.input_op_id);
        DynamicNodeInvocation relu1_invocation =
            dynamic_graph_get_invocation_for_id(tc.g, tc.relu1_op_id);
        DynamicNodeInvocation replicate_invocation =
            dynamic_graph_get_invocation_for_id(tc.g, tc.replicate_op_id);
        DynamicNodeInvocation relu2_invocation =
            dynamic_graph_get_invocation_for_id(tc.g, tc.relu2_op_id);

        DynamicNodeInvocation value_mapped_input_invocation =
            map_output_value(input_invocation, mk_single_shard_mapping(mc3));

        DynamicNodeInvocation value_mapped_relu1_invocation =
            map_input_and_output_values(relu1_invocation,
                                        mk_single_shard_mapping(mc1),
                                        mk_single_shard_mapping(mc1));

        DynamicNodeInvocation value_mapped_replicate_invocation =
            map_input_and_output_values(replicate_invocation,
                                        mk_single_shard_mapping(mc1),
                                        mk_two_shard_mapping(mc2, mc3));

        DynamicNodeInvocation value_mapped_relu2_invocation =
            map_input_and_output_values(relu2_invocation,
                                        mk_two_shard_mapping(mc1, mc3),
                                        mk_two_shard_mapping(mc1, mc3));

        DynamicNodeInvocation input_to_relu1_copy = DynamicNodeInvocation{
            /*inputs=*/{
                {
                    mk_slot(TensorSlotName::INPUT),
                    require_only_key(value_mapped_input_invocation.outputs,
                                     mk_slot(TensorSlotName::OUTPUT)),
                },
            },
            /*node_attrs=*/
            DynamicNodeAttrs{
                /*task_type=*/std::nullopt,
                /*device_ids=*/std::nullopt,
                /*mapping=*/std::nullopt,
                /*op_attrs=*/TrainingOperationAttrs{CopyAttrs{}},
                /*layer_guid=*/
                dynamic_layer_guid_t{dynamic_copy_layer_guid_t{}},
                /*per_device_op_state=*/std::nullopt,
            },
            /*outputs=*/
            {
                {
                    mk_slot(TensorSlotName::OUTPUT),
                    require_only_key(value_mapped_relu1_invocation.inputs,
                                     mk_slot(TensorSlotName::INPUT)),
                },
            },
        };

        DynamicNodeInvocation replicate_to_relu2_copy = DynamicNodeInvocation{
            /*inputs=*/{
                {
                    mk_slot(TensorSlotName::INPUT),
                    require_only_key(value_mapped_replicate_invocation.outputs,
                                     mk_slot(TensorSlotName::OUTPUT)),
                },
            },
            /*node_attrs=*/
            DynamicNodeAttrs{
                /*task_type=*/std::nullopt,
                /*device_ids=*/std::nullopt,
                /*mapping=*/std::nullopt,
                /*op_attrs=*/TrainingOperationAttrs{CopyAttrs{}},
                /*layer_guid=*/
                dynamic_layer_guid_t{dynamic_copy_layer_guid_t{}},
                /*per_device_op_state=*/std::nullopt,
            },
            /*outputs=*/
            {
                {
                    mk_slot(TensorSlotName::OUTPUT),
                    require_only_key(value_mapped_relu2_invocation.inputs,
                                     mk_slot(TensorSlotName::INPUT)),
                },
            },
        };

        return dynamic_open_dataflow_graph_from_invocation_set({
            value_mapped_input_invocation,
            input_to_relu1_copy,
            value_mapped_relu1_invocation,
            value_mapped_replicate_invocation,
            replicate_to_relu2_copy,
            value_mapped_relu2_invocation,
        });
      }();

      nlohmann::json result_json =
          dynamic_open_dataflow_graph_to_serializable(result);
      nlohmann::json correct_json =
          dynamic_open_dataflow_graph_to_serializable(correct);

      CHECK_MESSAGE(result == correct,
                    check_kv("result\n", result_json.dump()),
                    check_kv("correct\n", correct_json.dump()));
    }

    SUBCASE("copy insertion commutes with pass expansion") {
      MachineSpaceCoordinate mc1 = mk_machine_coord(0_n);
      MachineSpaceCoordinate mc2 = mk_machine_coord(1_n);
      MachineSpaceCoordinate mc3 = mk_machine_coord(2_n);

      SUBCASE("graph is single input node") {
        DynamicOpenDataflowGraph g = mk_single_input_node_graph(mc1);

        DynamicOpenDataflowGraph pass_expansion_before_copy_insertion =
            perform_copy_insertion(perform_pass_expansion(g));
        DynamicOpenDataflowGraph copy_insertion_before_pass_expansion =
            perform_pass_expansion(perform_copy_insertion(g));

        nlohmann::json pass_expansion_before_copy_insertion_json =
            dynamic_open_dataflow_graph_to_serializable(
                pass_expansion_before_copy_insertion);
        nlohmann::json copy_insertion_before_pass_expansion_json =
            dynamic_open_dataflow_graph_to_serializable(
                copy_insertion_before_pass_expansion);

        CHECK_MESSAGE(
            pass_expansion_before_copy_insertion ==
                copy_insertion_before_pass_expansion,
            check_kv("pass_expansion_before_copy_insertion_json\n",
                     pass_expansion_before_copy_insertion_json.dump()),
            check_kv("copy_insertion_before_pass_expansion_json\n",
                     copy_insertion_before_pass_expansion_json.dump()),
            check_kv("pass_expansion_before_copy_insertion\n",
                     dynamic_open_dataflow_graph_as_dot(
                         pass_expansion_before_copy_insertion)),
            check_kv("copy_insertion_before_pass_expansion\n",
                     dynamic_open_dataflow_graph_as_dot(
                         copy_insertion_before_pass_expansion)));
      }

      SUBCASE("graph is single input node followed by relu") {
        SUBCASE("operations are mapped to the same device") {
          DynamicOpenDataflowGraph g =
              mk_single_input_into_relu_graph(mc1, mc1);

          DynamicOpenDataflowGraph pass_expansion_before_copy_insertion =
              perform_copy_insertion(perform_pass_expansion(g));
          DynamicOpenDataflowGraph copy_insertion_before_pass_expansion =
              perform_pass_expansion(perform_copy_insertion(g));

          nlohmann::json pass_expansion_before_copy_insertion_json =
              dynamic_open_dataflow_graph_to_serializable(
                  pass_expansion_before_copy_insertion);
          nlohmann::json copy_insertion_before_pass_expansion_json =
              dynamic_open_dataflow_graph_to_serializable(
                  copy_insertion_before_pass_expansion);

          CHECK_MESSAGE(
              pass_expansion_before_copy_insertion ==
                  copy_insertion_before_pass_expansion,
              check_kv("pass_expansion_before_copy_insertion_json\n",
                       pass_expansion_before_copy_insertion_json.dump()),
              check_kv("copy_insertion_before_pass_expansion_json\n",
                       copy_insertion_before_pass_expansion_json.dump()),
              check_kv("pass_expansion_before_copy_insertion\n",
                       dynamic_open_dataflow_graph_as_dot(
                           pass_expansion_before_copy_insertion)),
              check_kv("copy_insertion_before_pass_expansion\n",
                       dynamic_open_dataflow_graph_as_dot(
                           copy_insertion_before_pass_expansion)));
        }

        SUBCASE("operations are mapped to different devices") {
          DynamicOpenDataflowGraph g =
              mk_single_input_into_relu_graph(mc1, mc2);

          DynamicOpenDataflowGraph pass_expansion_before_copy_insertion =
              perform_copy_insertion(perform_pass_expansion(g));
          DynamicOpenDataflowGraph copy_insertion_before_pass_expansion =
              perform_pass_expansion(perform_copy_insertion(g));

          nlohmann::json pass_expansion_before_copy_insertion_json =
              dynamic_open_dataflow_graph_to_serializable(
                  pass_expansion_before_copy_insertion);
          nlohmann::json copy_insertion_before_pass_expansion_json =
              dynamic_open_dataflow_graph_to_serializable(
                  copy_insertion_before_pass_expansion);

          CHECK_MESSAGE(
              pass_expansion_before_copy_insertion ==
                  copy_insertion_before_pass_expansion,
              check_kv("pass_expansion_before_copy_insertion_json\n",
                       pass_expansion_before_copy_insertion_json.dump()),
              check_kv("copy_insertion_before_pass_expansion_json\n",
                       copy_insertion_before_pass_expansion_json.dump()),
              check_kv("pass_expansion_before_copy_insertion\n",
                       dynamic_open_dataflow_graph_as_dot(
                           pass_expansion_before_copy_insertion)),
              check_kv("copy_insertion_before_pass_expansion\n",
                       dynamic_open_dataflow_graph_as_dot(
                           copy_insertion_before_pass_expansion)));
        }
      }

      SUBCASE("multinode graph including replicate") {
        ExampleGraphTestCase tc = mk_example_replicate_graph(
            /*input_device=*/mc3,
            /*relu1_device=*/mc1,
            /*replicate_device1=*/mc2,
            /*replicate_device2=*/mc3,
            /*relu2_device1=*/mc1,
            /*relu2_device2=*/mc3);

        DynamicOpenDataflowGraph pass_expansion_before_copy_insertion =
            perform_copy_insertion(perform_pass_expansion(tc.g));
        DynamicOpenDataflowGraph copy_insertion_before_pass_expansion =
            perform_pass_expansion(perform_copy_insertion(tc.g));

        nlohmann::json pass_expansion_before_copy_insertion_json =
            dynamic_open_dataflow_graph_to_serializable(
                pass_expansion_before_copy_insertion);
        nlohmann::json copy_insertion_before_pass_expansion_json =
            dynamic_open_dataflow_graph_to_serializable(
                copy_insertion_before_pass_expansion);

        CHECK_MESSAGE(
            pass_expansion_before_copy_insertion ==
                copy_insertion_before_pass_expansion,
            check_kv("pass_expansion_before_copy_insertion_json\n",
                     pass_expansion_before_copy_insertion_json.dump()),
            check_kv("copy_insertion_before_pass_expansion_json\n",
                     copy_insertion_before_pass_expansion_json.dump()),
            check_kv("pass_expansion_before_copy_insertion\n",
                     dynamic_open_dataflow_graph_as_dot(
                         pass_expansion_before_copy_insertion)),
            check_kv("copy_insertion_before_pass_expansion\n",
                     dynamic_open_dataflow_graph_as_dot(
                         copy_insertion_before_pass_expansion)));
      }
    }
  }
}
