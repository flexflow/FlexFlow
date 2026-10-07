#include "task-spec/dynamic_graph/machine_slicing.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("perform_machine_slicing_for_invocation") {
    auto mk_device_id = [](nonnegative_int node_idx,
                           nonnegative_int device_idx) -> global_device_id_t {
      return global_device_id_t{
          MachineSpaceCoordinate{
              /*node_idx=*/node_idx,
              /*device_idx=*/device_idx,
          },
          DeviceType::GPU,
      };
    };

    auto mk_pt_coord =
        [](nonnegative_int idx1,
           nonnegative_int idx2,
           nonnegative_int idx3,
           nonnegative_int idx4) -> ParallelTensorSpaceCoordinate {
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

    global_device_id_t mc1 = mk_device_id(0_n, 0_n);
    global_device_id_t mc2 = mk_device_id(2_n, 0_n);
    global_device_id_t mc3 = mk_device_id(4_n, 0_n);

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

    auto mk_slot = [](TensorSlotName const &slot_name) -> DynamicTensorSlot {
      return DynamicTensorSlot{
          /*slot_name=*/slot_name,
          /*slot_tensor_role=*/std::nullopt,
          /*task_shard=*/std::nullopt,
      };
    };

    auto mk_value =
        [](size_t src_node_id,
           TensorSlotName src_slot_name,
           std::optional<ParallelTensorSpaceCoordinate> const &shard_coord)
        -> DynamicValueAttrs {
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
          /*mapping=*/std::nullopt,
          /*accessor=*/std::nullopt,
          /*role=*/std::nullopt,
      };
    };

    size_t invocation1_id = 20;
    size_t invocation2_id = 21;
    size_t invocation3_id = 22;

    DynamicValueAttrs graph_input1 =
        mk_value(0, TensorSlotName::OUTPUT, std::nullopt);
    DynamicValueAttrs graph_input2 =
        mk_value(1, TensorSlotName::OUTPUT, std::nullopt);
    DynamicValueAttrs invocation1_output1 =
        mk_value(invocation1_id, TensorSlotName::OUTPUT_01, std::nullopt);
    DynamicValueAttrs invocation1_output2 =
        mk_value(invocation1_id, TensorSlotName::OUTPUT_02, std::nullopt);
    DynamicValueAttrs invocation2_output1 =
        mk_value(invocation2_id, TensorSlotName::OUTPUT_04, std::nullopt);
    DynamicValueAttrs invocation3_output1 =
        mk_value(invocation3_id, TensorSlotName::OUTPUT_01, std::nullopt);

    DynamicNodeInvocation invocation1 = DynamicNodeInvocation{
        /*inputs=*/{
            {
                mk_slot(TensorSlotName::INPUT),
                graph_input1,
            },
            {
                mk_slot(TensorSlotName::WEIGHT),
                graph_input2,
            },
        },
        /*node_attrs=*/
        DynamicNodeAttrs{
            /*task_type=*/std::nullopt,
            /*device_ids=*/nonempty_set{mc2},
            /*mapping=*/std::nullopt,
            /*op_attrs=*/std::nullopt,
            /*layer_guid=*/
            dynamic_layer_guid_t{parallel_layer_guid_t{Node{invocation1_id}}},
            /*per_device_op_state=*/std::nullopt,
        },
        /*outputs=*/
        {
            {
                mk_slot(TensorSlotName::OUTPUT_01),
                invocation1_output1,
            },
            {
                mk_slot(TensorSlotName::OUTPUT_02),
                invocation1_output2,
            },
        },
    };

    DynamicNodeInvocation invocation2 = DynamicNodeInvocation{
        /*inputs=*/{
            {mk_slot(TensorSlotName::INPUT), invocation1_output2},
        },
        /*node_attrs=*/
        DynamicNodeAttrs{
            /*task_type=*/std::nullopt,
            /*device_coord=*/nonempty_set{mc1},
            /*mapping=*/std::nullopt,
            /*op_attrs=*/std::nullopt,
            /*layer_guid=*/
            dynamic_layer_guid_t{parallel_layer_guid_t{Node{invocation2_id}}},
            /*per_device_op_state=*/std::nullopt,
        },
        /*outputs=*/
        {
            {
                mk_slot(TensorSlotName::OUTPUT_04),
                invocation2_output1,
            },
        },
    };

    DynamicNodeInvocation invocation3 = DynamicNodeInvocation{
        /*inputs=*/{
            {
                mk_slot(TensorSlotName::KEY),
                invocation2_output1,
            },
            {
                mk_slot(TensorSlotName::QUERY),
                graph_input2,
            },
            {
                mk_slot(TensorSlotName::VALUE),
                invocation1_output1,
            },
        },
        /*node_attrs=*/
        DynamicNodeAttrs{
            /*task_type=*/std::nullopt,
            /*device_coord=*/nonempty_set{mc2},
            /*mapping=*/std::nullopt,
            /*op_attrs=*/std::nullopt,
            /*layer_guid=*/
            dynamic_layer_guid_t{parallel_layer_guid_t{Node{invocation3_id}}},
            /*per_device_op_state=*/std::nullopt,
        },
        /*outputs=*/
        {
            {
                mk_slot(TensorSlotName::OUTPUT_01),
                invocation3_output1,
            },
        },
    };

    DynamicOpenDataflowGraph unsliced =
        dynamic_open_dataflow_graph_from_invocation_set(
            /*invocations=*/{
                invocation1,
                invocation2,
                invocation3,
            });

    SUBCASE("task exists on MachineCoord") {
      SUBCASE("mc1") {
        DynamicOpenDataflowGraph result =
            perform_machine_slicing(unsliced, mc1);

        DynamicOpenDataflowGraph correct =
            dynamic_open_dataflow_graph_from_invocation_set({
                invocation2,
            });

        CHECK(dynamic_open_dataflow_graphs_are_isomorphic(result, correct));
      }

      SUBCASE("mc2") {
        DynamicOpenDataflowGraph result =
            perform_machine_slicing(unsliced, mc2);

        DynamicOpenDataflowGraph correct =
            dynamic_open_dataflow_graph_from_invocation_set({
                invocation1,
                invocation3,
            });

        CHECK(dynamic_open_dataflow_graphs_are_isomorphic(result, correct));
      }
    }

    SUBCASE("task does not exist on MachineCoord") {
      DynamicOpenDataflowGraph result = perform_machine_slicing(unsliced, mc3);

      DynamicOpenDataflowGraph correct =
          dynamic_open_dataflow_graph_from_invocation_set(
              std::set<DynamicNodeInvocation>{});

      CHECK(dynamic_open_dataflow_graphs_are_isomorphic(result, correct));
    }
  }
}
