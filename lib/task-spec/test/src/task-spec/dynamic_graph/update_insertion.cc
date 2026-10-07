#include "task-spec/dynamic_graph/update_insertion.h"
#include "op-attrs/initializer_attrs.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_tensor_role.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("perform_update_insertion") {
    dynamic_layer_guid_t layer_guid = dynamic_layer_guid_t{
        parallel_layer_guid_t{
            Node{0},
        },
    };

    dynamic_tensor_guid_t tensor_guid = dynamic_tensor_guid_t{
        parallel_tensor_guid_t{
            KwargDataflowOutput<TensorSlotName>{
                /*node=*/Node{0},
                /*slot_name=*/TensorSlotName::OUTPUT,
            },
        },
    };

    TrainingOperationAttrs weight_attrs = TrainingOperationAttrs{
        PCGOperatorAttrs{
            WeightAttrs{
                /*tensor_shape=*/TensorShape{
                    /*dims=*/TensorDims{
                        FFOrdered{
                            4_p,
                            3_p,
                        },
                    },
                    /*data_type=*/DataType::FLOAT,
                },
                /*initializer=*/make_zero_initializer(),
            },
        },
    };

    DynamicNodeInvocation weight_invocation = DynamicNodeInvocation{
        /*inputs=*/{},
        /*node_attrs=*/
        DynamicNodeAttrs{
            /*task_type=*/DynamicTaskType::FWD,
            /*device_coord=*/std::nullopt,
            /*mapping=*/std::nullopt,
            /*op_attrs=*/weight_attrs,
            /*layer_guid=*/layer_guid,
            /*per_device_op_state=*/std::nullopt,
        },
        /*outputs=*/
        std::map<DynamicTensorSlot, DynamicValueAttrs>{
            {
                DynamicTensorSlot{
                    /*slot_name=*/TensorSlotName::OUTPUT,
                    /*slot_tensor_role=*/mk_dynamic_tensor_role_fwd(),
                    /*task_shard=*/std::nullopt,
                },
                DynamicValueAttrs{
                    /*tensor_guid=*/tensor_guid,
                    /*parallel_tensor_shape=*/std::nullopt,
                    /*create_grad=*/std::nullopt,
                    /*subgradient_id=*/std::nullopt,
                    /*shard_coord=*/std::nullopt,
                    /*mapping=*/std::nullopt,
                    /*accessor=*/std::nullopt,
                    /*role=*/mk_dynamic_tensor_role_fwd(),
                },
            },
        },
    };

    DynamicOpenDataflowGraph input =
        dynamic_open_dataflow_graph_from_invocation_set({weight_invocation});

    OptimizerAttrs optimizer_attrs = OptimizerAttrs{
        SGDOptimizerAttrs{
            /*lr=*/0.001,
            /*momentum=*/0.9,
            /*nesterov=*/false,
            /*weight_decay=*/0.001,
        },
    };

    DynamicOpenDataflowGraph result =
        perform_update_insertion(input, optimizer_attrs);

    DynamicOpenDataflowGraph correct =
        dynamic_open_dataflow_graph_from_invocation_set(
            /*invocations=*/{
                weight_invocation,
                DynamicNodeInvocation{
                    /*inputs=*/{
                        {
                            DynamicTensorSlot{
                                /*slot_name=*/TensorSlotName::OUTPUT,
                                /*slot_tensor_role=*/
                                mk_dynamic_tensor_role_fwd(),
                                /*task_shard=*/std::nullopt,
                            },
                            DynamicValueAttrs{
                                /*tensor_guid=*/tensor_guid,
                                /*parallel_tensor_shape=*/std::nullopt,
                                /*create_grad=*/std::nullopt,
                                /*subgradient_id=*/std::nullopt,
                                /*shard_coord=*/std::nullopt,
                                /*mapping=*/std::nullopt,
                                /*accessor=*/std::nullopt,
                                /*role=*/mk_dynamic_tensor_role_fwd(),
                            },
                        },
                        {
                            DynamicTensorSlot{
                                /*slot_name=*/TensorSlotName::OUTPUT,
                                /*slot_tensor_role=*/
                                mk_dynamic_tensor_role_bwd(),
                                /*task_shard=*/std::nullopt,
                            },
                            DynamicValueAttrs{
                                /*tensor_guid=*/tensor_guid,
                                /*parallel_tensor_shape=*/std::nullopt,
                                /*create_grad=*/std::nullopt,
                                /*subgradient_id=*/std::nullopt,
                                /*shard_coord=*/std::nullopt,
                                /*mapping=*/std::nullopt,
                                /*accessor=*/std::nullopt,
                                /*role=*/mk_dynamic_tensor_role_bwd(),
                            },
                        },
                        {
                            DynamicTensorSlot{
                                /*slot_name=*/TensorSlotName::OUTPUT,
                                /*slot_tensor_role=*/
                                mk_dynamic_tensor_role_opt(
                                    OptimizerSlotName::SGD_V),
                                /*task_shard=*/std::nullopt,
                            },
                            DynamicValueAttrs{
                                /*tensor_guid=*/tensor_guid,
                                /*parallel_tensor_shape=*/std::nullopt,
                                /*create_grad=*/std::nullopt,
                                /*subgradient_id=*/std::nullopt,
                                /*shard_coord=*/std::nullopt,
                                /*mapping=*/std::nullopt,
                                /*accessor=*/std::nullopt,
                                /*role=*/
                                mk_dynamic_tensor_role_opt(
                                    OptimizerSlotName::SGD_V),
                            },
                        },
                    },
                    /*node_attrs=*/
                    DynamicNodeAttrs{
                        /*task_type=*/DynamicTaskType::UPD,
                        /*device_coord=*/std::nullopt,
                        /*mapping=*/std::nullopt,
                        /*op_attrs=*/weight_attrs,
                        /*layer_guid=*/layer_guid,
                        /*per_device_op_state=*/std::nullopt,
                    },
                    /*outputs=*/{}},
            });

    CHECK(result == correct);
  }
}
