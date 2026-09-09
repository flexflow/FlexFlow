#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "op-attrs/initializer_attrs.h"
#include "task-spec/dynamic_graph/dynamic_tensor_role.h"
#include "task-spec/dynamic_graph/serializable_dynamic_value_attrs.h"
#include "utils/graph/instances/unordered_set_labelled_open_kwarg_dataflow_graph.h"
#include "utils/graph/node/algorithms.h"
#include <doctest/doctest.h>
#include <nlohmann/json.hpp>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("dynamic_op_dataflow_graph_from_invocation_set") {

    auto mk_dynamic_value = [](size_t node_id,
                               TensorSlotName slot_name) -> DynamicValueAttrs {
      return DynamicValueAttrs{
          /*tensor_guid=*/dynamic_tensor_guid_t{parallel_tensor_guid_t{
              KwargDataflowOutput{
                  Node{node_id},
                  slot_name,
              },
          }},
          /*parallel_tensor_shape=*/std::nullopt,
          /*create_grad=*/std::nullopt,
          /*subgradient_id=*/std::nullopt,
          /*shard_coord=*/std::nullopt,
          /*mapping=*/std::nullopt,
          /*accessor=*/std::nullopt,
          /*tensor_type=*/std::nullopt,
      };
    };

    auto mk_slot = [](TensorSlotName slot_name) {
      return DynamicTensorSlot{
          /*slot_name=*/slot_name,
          /*slot_tensor_role=*/std::nullopt,
          /*task_shard=*/std::nullopt,
      };
    };

    DynamicValueAttrs value_1 = mk_dynamic_value(1, TensorSlotName::OUTPUT);
    DynamicValueAttrs value_2 = mk_dynamic_value(2, TensorSlotName::OUTPUT);
    DynamicValueAttrs value_3 = mk_dynamic_value(3, TensorSlotName::OUTPUT);

    DynamicNodeAttrs node_attrs = DynamicNodeAttrs{
        /*task_type=*/std::nullopt,
        /*device_coord=*/std::nullopt,
        /*mapping=*/std::nullopt,
        /*op_attrs=*/std::nullopt,
        /*layer_guid=*/dynamic_layer_guid_t{parallel_layer_guid_t{Node{4}}},
        /*per_device_op_state=*/std::nullopt,
    };

    SUBCASE("correct usage") {
      DynamicNodeInvocation invocation_1 = DynamicNodeInvocation{
          /*inputs=*/std::map<DynamicTensorSlot, DynamicValueAttrs>{
              {
                  mk_slot(TensorSlotName::INPUT),
                  value_1,
              },
          },
          /*node_attrs=*/node_attrs,
          /*outputs=*/
          std::map<DynamicTensorSlot, DynamicValueAttrs>{
              {
                  mk_slot(TensorSlotName::OUTPUT),
                  value_2,
              },
          },
      };

      DynamicNodeInvocation invocation_2 = DynamicNodeInvocation{
          /*inputs=*/std::map<DynamicTensorSlot, DynamicValueAttrs>{},
          /*node_attrs=*/node_attrs,
          /*outputs=*/
          std::map<DynamicTensorSlot, DynamicValueAttrs>{
              {
                  mk_slot(TensorSlotName::OUTPUT),
                  value_3,
              },
          },
      };

      DynamicNodeInvocation invocation_3 = DynamicNodeInvocation{
          /*inputs=*/std::map<DynamicTensorSlot, DynamicValueAttrs>{
              {
                  mk_slot(TensorSlotName::INPUT),
                  value_1,
              },
              {
                  mk_slot(TensorSlotName::WEIGHT),
                  value_2,
              },
              {
                  mk_slot(TensorSlotName::BIAS),
                  value_1,
              },
          },
          /*node_attrs=*/node_attrs,
          /*outputs=*/
          std::map<DynamicTensorSlot, DynamicValueAttrs>{},
      };

      std::set<DynamicNodeInvocation> invocation_set = {
          invocation_1,
          invocation_2,
          invocation_3,
      };

      DynamicOpenDataflowGraph result =
          dynamic_open_dataflow_graph_from_invocation_set(invocation_set);

      CHECK(dynamic_graph_num_nodes(result) == 3);
    }

    SUBCASE("throws if multiple invocations produce the same value") {
      DynamicNodeInvocation invocation_1 = DynamicNodeInvocation{
          /*inputs=*/std::map<DynamicTensorSlot, DynamicValueAttrs>{
              {
                  mk_slot(TensorSlotName::INPUT),
                  value_1,
              },
          },
          /*node_attrs=*/node_attrs,
          /*outputs=*/
          std::map<DynamicTensorSlot, DynamicValueAttrs>{
              {
                  mk_slot(TensorSlotName::OUTPUT),
                  value_2,
              },
          },
      };

      DynamicNodeInvocation invocation_2 = DynamicNodeInvocation{
          /*inputs=*/std::map<DynamicTensorSlot, DynamicValueAttrs>{},
          /*node_attrs=*/node_attrs,
          /*outputs=*/
          std::map<DynamicTensorSlot, DynamicValueAttrs>{
              {
                  mk_slot(TensorSlotName::OUTPUT),
                  value_2,
              },
          },
      };

      std::set<DynamicNodeInvocation> invocation_set = {
          invocation_1,
          invocation_2,
      };

      CHECK_THROWS(
          dynamic_open_dataflow_graph_from_invocation_set(invocation_set));
    }

    SUBCASE("throws if invocations contain/create cycle") {
      DynamicNodeInvocation invocation_1 = DynamicNodeInvocation{
          /*inputs=*/std::map<DynamicTensorSlot, DynamicValueAttrs>{
              {
                  mk_slot(TensorSlotName::INPUT),
                  value_1,
              },
          },
          /*node_attrs=*/node_attrs,
          /*outputs=*/
          std::map<DynamicTensorSlot, DynamicValueAttrs>{
              {
                  mk_slot(TensorSlotName::OUTPUT),
                  value_2,
              },
          },
      };

      DynamicNodeInvocation invocation_2 = DynamicNodeInvocation{
          /*inputs=*/std::map<DynamicTensorSlot, DynamicValueAttrs>{
              {
                  mk_slot(TensorSlotName::INPUT),
                  value_2,
              },
          },
          /*node_attrs=*/node_attrs,
          /*outputs=*/
          std::map<DynamicTensorSlot, DynamicValueAttrs>{
              {
                  mk_slot(TensorSlotName::OUTPUT),
                  value_1,
              },
          },
      };

      std::set<DynamicNodeInvocation> invocation_set = {
          invocation_1,
          invocation_2,
      };

      CHECK_THROWS(
          dynamic_open_dataflow_graph_from_invocation_set(invocation_set));
    }
  }

  TEST_CASE("get_dynamic_slot_sites") {
    auto mk_dynamic_value = [](int node_id,
                               TensorSlotName slot_name) -> DynamicValueAttrs {
      return DynamicValueAttrs{
          /*tensor_guid=*/dynamic_tensor_guid_t{parallel_tensor_guid_t{
              KwargDataflowOutput{
                  Node{static_cast<size_t>(node_id)},
                  slot_name,
              },
          }},
          /*parallel_tensor_shape=*/std::nullopt,
          /*create_grad=*/std::nullopt,
          /*subgradient_id=*/std::nullopt,
          /*shard_coord=*/std::nullopt,
          /*mapping=*/std::nullopt,
          /*accessor=*/std::nullopt,
          /*tensor_type=*/std::nullopt,
      };
    };

    auto mk_node_attrs = [](int node_id) -> DynamicNodeAttrs {
      return DynamicNodeAttrs{
          /*task_type=*/std::nullopt,
          /*device_coord=*/std::nullopt,
          /*mapping=*/std::nullopt,
          /*op_attrs=*/std::nullopt,
          /*layer_guid=*/dynamic_layer_guid_t{parallel_layer_guid_t{Node{4}}},
          /*per_device_op_state=*/std::nullopt,
      };
    };

    DynamicValueAttrs value_1 = mk_dynamic_value(1, TensorSlotName::OUTPUT);
    DynamicValueAttrs value_2 = mk_dynamic_value(2, TensorSlotName::OUTPUT);
    DynamicValueAttrs value_3 = mk_dynamic_value(3, TensorSlotName::OUTPUT);

    DynamicNodeAttrs node_2 = mk_node_attrs(2);
    DynamicNodeAttrs node_3 = mk_node_attrs(3);
    DynamicNodeAttrs node_4 = mk_node_attrs(4);

    DynamicNodeInvocation invocation_1 = DynamicNodeInvocation{
        /*inputs=*/std::map<DynamicTensorSlot, DynamicValueAttrs>{
            {
                DynamicTensorSlot{
                    /*slot_name=*/TensorSlotName::INPUT,
                    /*slot_tensor_role=*/std::nullopt,
                    /*task_shard=*/std::nullopt,
                },
                value_1,
            },
        },
        /*node_attrs=*/node_2,
        /*outputs=*/
        std::map<DynamicTensorSlot, DynamicValueAttrs>{
            {
                DynamicTensorSlot{
                    /*slot_name=*/TensorSlotName::OUTPUT,
                    /*slot_tensor_role=*/std::nullopt,
                    /*task_shard=*/std::nullopt,
                },
                value_2,
            },
        },
    };

    DynamicNodeInvocation invocation_2 = DynamicNodeInvocation{
        /*inputs=*/std::map<DynamicTensorSlot, DynamicValueAttrs>{},
        /*node_attrs=*/node_3,
        /*outputs=*/
        std::map<DynamicTensorSlot, DynamicValueAttrs>{
            {
                DynamicTensorSlot{
                    /*slot_name=*/TensorSlotName::OUTPUT,
                    /*slot_tensor_role=*/std::nullopt,
                    /*task_shard=*/std::nullopt,
                },
                value_3,
            },
        },
    };

    DynamicNodeInvocation invocation_3 = DynamicNodeInvocation{
        /*inputs=*/std::map<DynamicTensorSlot, DynamicValueAttrs>{
            {
                DynamicTensorSlot{
                    /*slot_name=*/TensorSlotName::INPUT,
                    /*slot_tensor_role=*/std::nullopt,
                    /*task_shard=*/std::nullopt,
                },
                value_2,
            },
            {
                DynamicTensorSlot{
                    /*slot_name=*/TensorSlotName::WEIGHT,
                    /*slot_tensor_role=*/std::nullopt,
                    /*task_shard=*/std::nullopt,
                },
                value_2,
            },
            {
                DynamicTensorSlot{
                    /*slot_name=*/TensorSlotName::BIAS,
                    /*slot_tensor_role=*/std::nullopt,
                    /*task_shard=*/std::nullopt,
                },
                value_1,
            },
        },
        /*node_attrs=*/node_4,
        /*outputs=*/std::map<DynamicTensorSlot, DynamicValueAttrs>{},
    };

    std::set<DynamicNodeInvocation> invocation_set = {
        invocation_1,
        invocation_2,
        invocation_3,
    };

    DynamicOpenDataflowGraph g =
        dynamic_open_dataflow_graph_from_invocation_set(invocation_set);

    dynamic_invocation_id_t invocation_1_id =
        dynamic_graph_get_id_for_invocation(g, invocation_1);
    dynamic_invocation_id_t invocation_2_id =
        dynamic_graph_get_id_for_invocation(g, invocation_2);
    dynamic_invocation_id_t invocation_3_id =
        dynamic_graph_get_id_for_invocation(g, invocation_3);

    dynamic_value_id_t value_1_id = dynamic_graph_get_id_for_value(g, value_1);

    std::set<DynamicSlotSite> result = get_dynamic_slot_sites(g);

    auto mk_internal_slot_site =
        [](dynamic_invocation_id_t const &invocation_id,
           TensorDirection direction,
           TensorSlotName slot_name) -> DynamicSlotSite {
      return DynamicSlotSite{
          InternalDynamicSlotSite{
              /*invocation_id=*/invocation_id,
              /*direction=*/direction,
              /*slot_name=*/
              DynamicTensorSlot{
                  /*slot_name=*/slot_name,
                  /*slot_tensor_role=*/std::nullopt,
                  /*task_shard=*/std::nullopt,
              },
          },
      };
    };

    std::set<DynamicSlotSite> correct = {
        DynamicSlotSite{
            ExternalDynamicSlotSite{
                value_1_id.require_external(),
            },
        },
        mk_internal_slot_site(
            invocation_1_id, TensorDirection::INCOMING, TensorSlotName::INPUT),
        mk_internal_slot_site(
            invocation_1_id, TensorDirection::OUTPUT, TensorSlotName::OUTPUT),
        mk_internal_slot_site(
            invocation_2_id, TensorDirection::OUTPUT, TensorSlotName::OUTPUT),
        mk_internal_slot_site(
            invocation_3_id, TensorDirection::INCOMING, TensorSlotName::INPUT),
        mk_internal_slot_site(
            invocation_3_id, TensorDirection::INCOMING, TensorSlotName::WEIGHT),
        mk_internal_slot_site(
            invocation_3_id, TensorDirection::INCOMING, TensorSlotName::BIAS),
    };

    nlohmann::json result_json = result;
    nlohmann::json correct_json = correct;

    CHECK(result_json == correct_json);
  }

  TEST_CASE(
      "labelled_open_kwarg_dataflow_graph_from_dynamic_open_dataflow_graph") {
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

    DynamicNodeAttrs fwd_weight_node_attrs = DynamicNodeAttrs{
        /*task_type=*/DynamicTaskType::FWD,
        /*device_coord=*/std::nullopt,
        /*mapping=*/std::nullopt,
        /*op_attrs=*/weight_attrs,
        /*layer_guid=*/layer_guid,
        /*per_device_op_state=*/std::nullopt,
    };

    DynamicTensorSlot fwd_weight_output_slot1 = DynamicTensorSlot{
        /*slot_name=*/TensorSlotName::OUTPUT,
        /*slot_tensor_role=*/mk_dynamic_tensor_role_fwd(),
        /*task_shard=*/std::nullopt,
    };

    DynamicValueAttrs fwd_weight_output_attrs1 = DynamicValueAttrs{
        /*tensor_guid=*/tensor_guid,
        /*parallel_tensor_shape=*/std::nullopt,
        /*create_grad=*/std::nullopt,
        /*subgradient_id=*/std::nullopt,
        /*shard_coord=*/std::nullopt,
        /*mapping=*/std::nullopt,
        /*accessor=*/std::nullopt,
        /*role=*/mk_dynamic_tensor_role_fwd(),
    };

    DynamicNodeInvocation weight_invocation = DynamicNodeInvocation{
        /*inputs=*/{},
        /*node_attrs=*/fwd_weight_node_attrs,
        /*outputs=*/
        std::map<DynamicTensorSlot, DynamicValueAttrs>{
            {
                fwd_weight_output_slot1,
                fwd_weight_output_attrs1,
            },
        },
    };

    DynamicNodeAttrs upd_weight_node_attrs = DynamicNodeAttrs{
        /*task_type=*/DynamicTaskType::UPD,
        /*device_coord=*/std::nullopt,
        /*mapping=*/std::nullopt,
        /*op_attrs=*/weight_attrs,
        /*layer_guid=*/layer_guid,
        /*per_device_op_state=*/std::nullopt,
    };

    DynamicTensorSlot upd_weight_input_slot2 = DynamicTensorSlot{
        /*slot_name=*/TensorSlotName::OUTPUT,
        /*slot_tensor_role=*/mk_dynamic_tensor_role_bwd(),
        /*task_shard=*/std::nullopt,
    };

    DynamicValueAttrs upd_weight_input_attrs2 = DynamicValueAttrs{
        /*tensor_guid=*/tensor_guid,
        /*parallel_tensor_shape=*/std::nullopt,
        /*create_grad=*/std::nullopt,
        /*subgradient_id=*/std::nullopt,
        /*shard_coord=*/std::nullopt,
        /*mapping=*/std::nullopt,
        /*accessor=*/std::nullopt,
        /*role=*/mk_dynamic_tensor_role_bwd(),
    };

    DynamicTensorSlot upd_weight_input_slot3 = DynamicTensorSlot{
        /*slot_name=*/TensorSlotName::OUTPUT,
        /*slot_tensor_role=*/
        mk_dynamic_tensor_role_opt(OptimizerSlotName::SGD_V),
        /*task_shard=*/std::nullopt,
    };

    DynamicValueAttrs upd_weight_input_attrs3 = DynamicValueAttrs{
        /*tensor_guid=*/tensor_guid,
        /*parallel_tensor_shape=*/std::nullopt,
        /*create_grad=*/std::nullopt,
        /*subgradient_id=*/std::nullopt,
        /*shard_coord=*/std::nullopt,
        /*mapping=*/std::nullopt,
        /*accessor=*/std::nullopt,
        /*role=*/mk_dynamic_tensor_role_opt(OptimizerSlotName::SGD_V),
    };

    DynamicOpenDataflowGraph input =
        dynamic_open_dataflow_graph_from_invocation_set(
            /*invocations=*/{
                weight_invocation,
                DynamicNodeInvocation{/*inputs=*/{
                                          {
                                              fwd_weight_output_slot1,
                                              fwd_weight_output_attrs1,
                                          },
                                          {
                                              upd_weight_input_slot2,
                                              upd_weight_input_attrs2,
                                          },
                                          {
                                              upd_weight_input_slot3,
                                              upd_weight_input_attrs3,
                                          },
                                      },
                                      /*node_attrs=*/upd_weight_node_attrs,
                                      /*outputs=*/{}},
            });

    std::pair<LabelledOpenKwargDataflowGraph<DynamicNodeAttrs,
                                             DynamicValueAttrs,
                                             int,
                                             DynamicTensorSlot>,
              bidict<Node, DynamicNodeInvocation>>
        result =
            labelled_open_kwarg_dataflow_graph_from_dynamic_open_dataflow_graph(
                input);

    LabelledOpenKwargDataflowGraph<DynamicNodeAttrs,
                                   DynamicValueAttrs,
                                   int,
                                   DynamicTensorSlot>
        correct = LabelledOpenKwargDataflowGraph<DynamicNodeAttrs,
                                                 DynamicValueAttrs,
                                                 int,
                                                 DynamicTensorSlot>::
            create<UnorderedSetLabelledOpenKwargDataflowGraph<
                DynamicNodeAttrs,
                DynamicValueAttrs,
                int,
                DynamicTensorSlot>>();

    KwargNodeAddedResult<DynamicTensorSlot> fwd_weight_added = correct.add_node(
        /*node_label=*/fwd_weight_node_attrs,
        /*inputs=*/{},
        /*output_labels=*/
        {
            {
                fwd_weight_output_slot1,
                fwd_weight_output_attrs1,
            },
        });

    KwargNodeAddedResult<DynamicTensorSlot> upd_weight_added = correct.add_node(
        /*node_label=*/fwd_weight_node_attrs,
        /*inputs=*/{},
        /*output_labels=*/
        {
            {
                fwd_weight_output_slot1,
                fwd_weight_output_attrs1,
            },
        });
  }
}
