#include "task-spec/dynamic_graph/pass_expansion.h"
#include "op-attrs/initializer_attrs.h"
#include "op-attrs/ops/element_unary.h"
#include "task-spec/dynamic_graph/dynamic_invocation_id_t.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_tensor_role.h"
#include "task-spec/dynamic_graph/serializable_dynamic_node_invocation.h"
#include "task-spec/dynamic_graph/serializable_dynamic_open_dataflow_graph.h"
#include "test/utils/doctest/check_kv.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("determine_intermediate_values_needed_for_gradient_computation") {
    auto mk_slot = [](TensorSlotName slot_name) -> DynamicTensorSlot {
      return DynamicTensorSlot{
          /*slot_name=*/slot_name,
          /*slot_tensor_role=*/std::nullopt,
          /*task_shard=*/std::nullopt,
      };
    };

    auto mk_node_attrs = [](size_t layer_guid,
                            PCGOperatorAttrs const &op_attrs) {
      return DynamicNodeAttrs{
          /*task_type=*/std::nullopt,
          /*device_ids=*/std::nullopt,
          /*mapping=*/std::nullopt,
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
    };

    auto mk_value_attrs = [](size_t src_layer_guid,
                             TensorSlotName src_slot,
                             bool create_grad) -> DynamicValueAttrs {
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
          /*create_grad=*/create_grad,
          /*subgradient_id=*/std::nullopt,
          /*shard_coord=*/std::nullopt,
          /*mapping=*/std::nullopt,
          /*accessor=*/std::nullopt,
          /*role=*/std::nullopt,
      };
    };

    struct TestGraph {
      DynamicOpenDataflowGraph g;
      DynamicValueAttrs input_op_output;
      DynamicValueAttrs relu1_op_output;
    };

    auto mk_test_graph = [&](bool input_create_grad) -> TestGraph {
      TensorShape input_shape = TensorShape{
          TensorDims{
              FFOrdered<positive_int>{
                  8_p,
                  5_p,
              },
          },
          DataType::FLOAT,
      };

      DynamicValueAttrs input_op_output = mk_value_attrs(
          123, TensorSlotName::OUTPUT, /*create_grad=*/input_create_grad);

      DynamicValueAttrs relu1_op_output =
          mk_value_attrs(124, TensorSlotName::OUTPUT, /*create_grad=*/true);

      PCGOperatorAttrs input_attrs = PCGOperatorAttrs{
          InputAttrs{
              input_shape,
          },
      };

      PCGOperatorAttrs relu_attrs = PCGOperatorAttrs{
          make_relu_attrs(),
      };

      DynamicNodeInvocation input_invocation = DynamicNodeInvocation{
          /*inputs=*/{},
          /*node_attrs=*/
          mk_node_attrs(
              /*layer_guid=*/123,
              /*op_attrs=*/PCGOperatorAttrs{InputAttrs{input_shape}}),
          /*outputs=*/
          {
              {
                  mk_slot(TensorSlotName::OUTPUT),
                  input_op_output,
              },
          },
      };

      DynamicNodeInvocation relu_invocation = DynamicNodeInvocation{
          /*inputs=*/{
              {
                  mk_slot(TensorSlotName::INPUT),
                  input_op_output,
              },
          },
          /*node_attrs=*/
          mk_node_attrs(
              /*layer_guid=*/124,
              /*op_attrs=*/relu_attrs),
          /*outputs=*/
          {
              {
                  mk_slot(TensorSlotName::OUTPUT),
                  relu1_op_output,
              },
          },
      };

      DynamicOpenDataflowGraph g =
          dynamic_open_dataflow_graph_from_invocation_set(
              {input_invocation, relu_invocation});

      return TestGraph{
          /*g=*/g,
          /*input_op_output=*/input_op_output,
          /*relu1_op_output=*/relu1_op_output,
      };
    };

    SUBCASE("input create_grad is false") {
      TestGraph tg = mk_test_graph(/*input_create_grad=*/false);

      std::set<DynamicValueAttrs> result =
          determine_intermediate_values_needed_for_gradient_computation(tg.g);

      std::set<DynamicValueAttrs> correct = {};

      ASSERT(result == correct);
    }

    SUBCASE("input create_grad is true") {
      TestGraph tg = mk_test_graph(/*input_create_grad=*/true);

      std::set<DynamicValueAttrs> result =
          determine_intermediate_values_needed_for_gradient_computation(tg.g);

      std::set<DynamicValueAttrs> correct = {
          tg.input_op_output,
          tg.relu1_op_output,
      };

      ASSERT(result == correct);
    }
  }

  TEST_CASE("perform_fwd_pass_expansion_for_invocation") {
    auto mk_value_attrs =
        [](size_t node_id, std::optional<DynamicTensorRole> const &tensor_role)
        -> DynamicValueAttrs {
      return DynamicValueAttrs{
          /*tensor_guid=*/dynamic_tensor_guid_t{parallel_tensor_guid_t{
              KwargDataflowOutput<TensorSlotName>{
                  Node{node_id},
                  TensorSlotName::OUTPUT,
              },
          }},
          /*parallel_tensor_shape=*/std::nullopt,
          /*create_grad=*/std::nullopt,
          /*subgradient_id=*/std::nullopt,
          /*shard_coord=*/std::nullopt,
          /*mapping=*/std::nullopt,
          /*accessor=*/std::nullopt,
          /*role=*/tensor_role,
      };
    };

    auto mk_slot =
        [](TensorSlotName const &slot_name,
           std::optional<DynamicTensorRole> role) -> DynamicTensorSlot {
      return DynamicTensorSlot{
          /*slot_name=*/slot_name,
          /*slot_tensor_role=*/role,
          /*task_shard=*/std::nullopt,
      };
    };

    dynamic_layer_guid_t layer_guid{parallel_layer_guid_t{Node{20}}};

    DynamicValueAttrs v1 = mk_value_attrs(0, std::nullopt);
    DynamicValueAttrs v2 = mk_value_attrs(1, std::nullopt);
    DynamicValueAttrs v3 = mk_value_attrs(2, std::nullopt);

    DynamicTensorRole fwd_role = DynamicTensorRole{FwbTensorType::FORWARD};

    DynamicValueAttrs v1_fwd = mk_value_attrs(0, fwd_role);
    DynamicValueAttrs v2_fwd = mk_value_attrs(1, fwd_role);
    DynamicValueAttrs v3_fwd = mk_value_attrs(2, fwd_role);

    SUBCASE("standard operator") {
      TrainingOperationAttrs op_attrs = TrainingOperationAttrs{
          PCGOperatorAttrs{
              LinearAttrs{
                  /*out_channels=*/8_p,
                  /*use_bias=*/true,
                  /*data_type=*/DataType::FLOAT,
                  /*activation=*/std::nullopt,
                  /*regularizer=*/std::nullopt,
              },
          },
      };

      DynamicNodeInvocation invocation = [&]() -> DynamicNodeInvocation {
        return DynamicNodeInvocation{
            /*inputs=*/{
                {mk_slot(TensorSlotName::INPUT, std::nullopt), v1},
                {mk_slot(TensorSlotName::WEIGHT, std::nullopt), v2},
                {mk_slot(TensorSlotName::BIAS, std::nullopt), v1},
            },
            /*node_attrs=*/
            DynamicNodeAttrs{
                /*task_type=*/std::nullopt,
                /*device_coord=*/std::nullopt,
                /*mapping=*/std::nullopt,
                /*op_attrs=*/op_attrs,
                /*layer_guid=*/layer_guid,
                /*per_device_op_state=*/std::nullopt,
            },
            /*outputs=*/
            {
                {mk_slot(TensorSlotName::OUTPUT, std::nullopt), v3},
            },
        };
      }();

      DynamicNodeInvocation result =
          perform_fwd_pass_expansion_for_invocation(invocation);

      DynamicNodeInvocation correct = DynamicNodeInvocation{
          /*inputs=*/{
              {mk_slot(TensorSlotName::INPUT, fwd_role), v1_fwd},
              {mk_slot(TensorSlotName::WEIGHT, fwd_role), v2_fwd},
              {mk_slot(TensorSlotName::BIAS, fwd_role), v1_fwd},
          },
          /*node_attrs=*/
          DynamicNodeAttrs{
              /*task_type=*/DynamicTaskType::FWD,
              /*device_coord=*/std::nullopt,
              /*mapping=*/std::nullopt,
              /*op_attrs=*/op_attrs,
              /*layer_guid=*/layer_guid,
              /*per_device_op_state=*/std::nullopt,
          },
          /*outputs=*/
          {
              {mk_slot(TensorSlotName::OUTPUT, fwd_role), v3_fwd},
          },
      };

      ASSERT(result == correct);
    }

    SUBCASE("copy operator") {
      TrainingOperationAttrs op_attrs = TrainingOperationAttrs{CopyAttrs{}};

      DynamicNodeInvocation invocation = [&]() -> DynamicNodeInvocation {
        return DynamicNodeInvocation{
            /*inputs=*/{
                {mk_slot(TensorSlotName::INPUT, std::nullopt), v1},
            },
            /*node_attrs=*/
            DynamicNodeAttrs{
                /*task_type=*/std::nullopt,
                /*device_coord=*/std::nullopt,
                /*mapping=*/std::nullopt,
                /*op_attrs=*/op_attrs,
                /*layer_guid=*/layer_guid,
                /*per_device_op_state=*/std::nullopt,
            },
            /*outputs=*/
            {
                {mk_slot(TensorSlotName::OUTPUT, std::nullopt), v2},
            },
        };
      }();

      DynamicNodeInvocation result =
          perform_fwd_pass_expansion_for_invocation(invocation);

      DynamicNodeInvocation correct = DynamicNodeInvocation{
          /*inputs=*/{
              {mk_slot(TensorSlotName::INPUT, std::nullopt), v1_fwd},
          },
          /*node_attrs=*/
          DynamicNodeAttrs{
              /*task_type=*/std::nullopt,
              /*device_coord=*/std::nullopt,
              /*mapping=*/std::nullopt,
              /*op_attrs=*/op_attrs,
              /*layer_guid=*/layer_guid,
              /*per_device_op_state=*/std::nullopt,
          },
          /*outputs=*/
          {
              {mk_slot(TensorSlotName::OUTPUT, std::nullopt), v2_fwd},
          },
      };

      ASSERT(dynamic_node_invocation_to_serializable(result) ==
             dynamic_node_invocation_to_serializable(correct));
    }
  }

  TEST_CASE("perform_bwd_pass_expansion_for_invocation") {
    auto mk_value_attrs =
        [](size_t node_id, std::optional<DynamicTensorRole> const &tensor_role)
        -> DynamicValueAttrs {
      return DynamicValueAttrs{
          /*tensor_guid=*/dynamic_tensor_guid_t{parallel_tensor_guid_t{
              KwargDataflowOutput<TensorSlotName>{
                  Node{node_id},
                  TensorSlotName::OUTPUT,
              },
          }},
          /*parallel_tensor_shape=*/std::nullopt,
          /*create_grad=*/std::nullopt,
          /*subgradient_id=*/std::nullopt,
          /*shard_coord=*/std::nullopt,
          /*mapping=*/std::nullopt,
          /*accessor=*/std::nullopt,
          /*role=*/tensor_role,
      };
    };

    auto mk_slot =
        [](TensorSlotName const &slot_name,
           std::optional<DynamicTensorRole> role) -> DynamicTensorSlot {
      return DynamicTensorSlot{
          /*slot_name=*/slot_name,
          /*slot_tensor_role=*/role,
          /*task_shard=*/std::nullopt,
      };
    };

    dynamic_layer_guid_t layer_guid{parallel_layer_guid_t{Node{20}}};

    DynamicValueAttrs v1 = mk_value_attrs(0, std::nullopt);
    DynamicValueAttrs v2 = mk_value_attrs(1, std::nullopt);
    DynamicValueAttrs v3 = mk_value_attrs(2, std::nullopt);

    DynamicTensorRole fwd_role = DynamicTensorRole{FwbTensorType::FORWARD};
    DynamicTensorRole grad_role = DynamicTensorRole{FwbTensorType::GRADIENT};

    DynamicValueAttrs v1_fwd = mk_value_attrs(0, fwd_role);
    DynamicValueAttrs v2_fwd = mk_value_attrs(1, fwd_role);
    DynamicValueAttrs v3_fwd = mk_value_attrs(2, fwd_role);
    DynamicValueAttrs v1_grad = mk_value_attrs(0, grad_role);
    DynamicValueAttrs v2_grad = mk_value_attrs(1, grad_role);
    DynamicValueAttrs v3_grad = mk_value_attrs(2, grad_role);

    SUBCASE("normal operator") {
      TrainingOperationAttrs op_attrs = TrainingOperationAttrs{
          PCGOperatorAttrs{
              LinearAttrs{
                  /*out_channels=*/8_p,
                  /*use_bias=*/true,
                  /*data_type=*/DataType::FLOAT,
                  /*activation=*/std::nullopt,
                  /*regularizer=*/std::nullopt,
              },
          },
      };

      dynamic_invocation_id_t invocation_id{0_n};

      DynamicNodeInvocation invocation = [&]() -> DynamicNodeInvocation {
        return DynamicNodeInvocation{
            /*inputs=*/{
                {mk_slot(TensorSlotName::INPUT, std::nullopt), v1},
                {mk_slot(TensorSlotName::WEIGHT, std::nullopt), v2},
                {mk_slot(TensorSlotName::BIAS, std::nullopt), v1},
            },
            /*node_attrs=*/
            DynamicNodeAttrs{
                /*task_type=*/std::nullopt,
                /*device_coord=*/std::nullopt,
                /*mapping=*/std::nullopt,
                /*op_attrs=*/op_attrs,
                /*layer_guid=*/layer_guid,
                /*per_device_op_state=*/std::nullopt,
            },
            /*outputs=*/
            {
                {mk_slot(TensorSlotName::OUTPUT, std::nullopt), v3},
            },
        };
      }();

      DynamicNodeInvocation result = perform_bwd_pass_expansion_for_invocation(
          invocation_id, invocation, {});

      DynamicNodeInvocation correct = [&]() -> DynamicNodeInvocation {
        return DynamicNodeInvocation{
            /*inputs=*/{
                {mk_slot(TensorSlotName::INPUT, fwd_role), v1_fwd},
                {mk_slot(TensorSlotName::WEIGHT, fwd_role), v2_fwd},
                {mk_slot(TensorSlotName::BIAS, fwd_role), v1_fwd},
                {mk_slot(TensorSlotName::OUTPUT, fwd_role), v3_fwd},
                {mk_slot(TensorSlotName::OUTPUT, grad_role), v3_grad},
            },
            /*node_attrs=*/
            DynamicNodeAttrs{
                /*pass_type=*/DynamicTaskType::BWD,
                /*device_coord=*/std::nullopt,
                /*mapping=*/std::nullopt,
                /*op_attrs=*/op_attrs,
                /*layer_guid=*/layer_guid,
                /*per_device_op_state=*/std::nullopt,
            },
            /*outputs=*/
            {
                {mk_slot(TensorSlotName::INPUT, grad_role), v1_grad},
                {mk_slot(TensorSlotName::WEIGHT, grad_role), v2_grad},
                {mk_slot(TensorSlotName::BIAS, grad_role), v1_grad},
            },
        };
      }();

      ASSERT(dynamic_node_invocation_to_serializable(result) ==
             dynamic_node_invocation_to_serializable(correct));
    }

    SUBCASE("replicate operator") {
      TrainingOperationAttrs op_attrs = TrainingOperationAttrs{
          PCGOperatorAttrs{
              ReplicateAttrs{
                  /*replicate_degree=*/2_p,
              },
          },
      };

      dynamic_invocation_id_t invocation_id{0_n};

      DynamicNodeInvocation invocation = [&]() -> DynamicNodeInvocation {
        return DynamicNodeInvocation{
            /*inputs=*/{
                {mk_slot(TensorSlotName::INPUT, std::nullopt), v1},
            },
            /*node_attrs=*/
            DynamicNodeAttrs{
                /*task_type=*/std::nullopt,
                /*device_coord=*/std::nullopt,
                /*mapping=*/std::nullopt,
                /*op_attrs=*/op_attrs,
                /*layer_guid=*/layer_guid,
                /*per_device_op_state=*/std::nullopt,
            },
            /*outputs=*/
            {
                {mk_slot(TensorSlotName::OUTPUT, std::nullopt), v2},
            },
        };
      }();

      DynamicNodeInvocation result = perform_bwd_pass_expansion_for_invocation(
          invocation_id, invocation, {});

      DynamicNodeInvocation correct = [&]() -> DynamicNodeInvocation {
        DynamicTensorRole fwd_role = DynamicTensorRole{FwbTensorType::FORWARD};
        DynamicTensorRole grad_role =
            DynamicTensorRole{FwbTensorType::GRADIENT};

        return DynamicNodeInvocation{
            /*inputs=*/{
                {mk_slot(TensorSlotName::OUTPUT, grad_role), v2_grad},
            },
            /*node_attrs=*/
            DynamicNodeAttrs{
                /*pass_type=*/DynamicTaskType::BWD,
                /*device_coord=*/std::nullopt,
                /*mapping=*/std::nullopt,
                /*op_attrs=*/op_attrs,
                /*layer_guid=*/layer_guid,
                /*per_device_op_state=*/std::nullopt,
            },
            /*outputs=*/
            {
                {mk_slot(TensorSlotName::INPUT, grad_role), v1_grad},
            },
        };
      }();

      ASSERT(dynamic_node_invocation_to_serializable(result) ==
             dynamic_node_invocation_to_serializable(correct));
    }

    SUBCASE("copy operator") {
      TrainingOperationAttrs op_attrs = TrainingOperationAttrs{CopyAttrs{}};

      dynamic_invocation_id_t invocation_id{0_n};

      DynamicNodeInvocation invocation = [&]() -> DynamicNodeInvocation {
        return DynamicNodeInvocation{
            /*inputs=*/{
                {mk_slot(TensorSlotName::INPUT, std::nullopt), v1},
            },
            /*node_attrs=*/
            DynamicNodeAttrs{
                /*task_type=*/std::nullopt,
                /*device_coord=*/std::nullopt,
                /*mapping=*/std::nullopt,
                /*op_attrs=*/op_attrs,
                /*layer_guid=*/layer_guid,
                /*per_device_op_state=*/std::nullopt,
            },
            /*outputs=*/
            {
                {mk_slot(TensorSlotName::OUTPUT, std::nullopt), v2},
            },
        };
      }();

      DynamicNodeInvocation result = perform_bwd_pass_expansion_for_invocation(
          invocation_id, invocation, {});

      DynamicNodeInvocation correct = [&]() -> DynamicNodeInvocation {
        DynamicTensorRole fwd_role = DynamicTensorRole{FwbTensorType::FORWARD};
        DynamicTensorRole grad_role =
            DynamicTensorRole{FwbTensorType::GRADIENT};

        return DynamicNodeInvocation{
            /*inputs=*/{
                {mk_slot(TensorSlotName::OUTPUT, std::nullopt), v2_grad},
            },
            /*node_attrs=*/
            DynamicNodeAttrs{
                /*pass_type=*/std::nullopt,
                /*device_coord=*/std::nullopt,
                /*mapping=*/std::nullopt,
                /*op_attrs=*/op_attrs,
                /*layer_guid=*/layer_guid,
                /*per_device_op_state=*/std::nullopt,
            },
            /*outputs=*/
            {
                {mk_slot(TensorSlotName::INPUT, std::nullopt), v1_grad},
            },
        };
      }();

      ASSERT(dynamic_node_invocation_to_serializable(result) ==
             dynamic_node_invocation_to_serializable(correct));
    }
  }

  TEST_CASE("perform_pass_expansion(DynamicOpenDataflowGraph)") {
    auto mk_node_attrs = [](size_t layer_id,
                            TrainingOperationAttrs const &op_attrs,
                            std::optional<DynamicTaskType> const &pass_type)
        -> DynamicNodeAttrs {
      return DynamicNodeAttrs{
          /*pass_type=*/pass_type,
          /*device_coord=*/std::nullopt,
          /*mapping=*/std::nullopt,
          /*op_attrs=*/op_attrs,
          /*layer_guid=*/
          dynamic_layer_guid_t{parallel_layer_guid_t{Node{layer_id}}},
          /*per_device_op_state=*/std::nullopt,
      };
    };

    auto mk_value_attrs =
        [](size_t node_id, std::optional<DynamicTensorRole> const &tensor_type)
        -> DynamicValueAttrs {
      return DynamicValueAttrs{
          /*tensor_guid=*/dynamic_tensor_guid_t{parallel_tensor_guid_t{
              KwargDataflowOutput{
                  Node{node_id},
                  TensorSlotName::OUTPUT,
              },
          }},
          /*parallel_tensor_shape=*/std::nullopt,
          /*create_grad=*/false,
          /*subgradient_id=*/std::nullopt,
          /*shard_coord=*/std::nullopt,
          /*mapping=*/std::nullopt,
          /*accessor=*/std::nullopt,
          /*role=*/tensor_type,
      };
    };

    TrainingOperationAttrs input_op_attrs = TrainingOperationAttrs{
        PCGOperatorAttrs{
            InputAttrs{
                TensorShape{
                    TensorDims{
                        FFOrdered<positive_int>{
                            4_p,
                            8_p,
                        },
                    },
                    DataType::FLOAT,
                },
            },
        },
    };

    TrainingOperationAttrs relu_op_attrs = TrainingOperationAttrs{
        PCGOperatorAttrs{
            make_relu_attrs(),
        },
    };

    TensorShape weight_shape = TensorShape{
        TensorDims{
            FFOrdered<positive_int>{
                8_p,
                6_p,
            },
        },
        DataType::FLOAT,
    };

    TrainingOperationAttrs weight_op_attrs = TrainingOperationAttrs{
        PCGOperatorAttrs{
            WeightAttrs{
                /*tensor_shape=*/weight_shape,
                /*initializer=*/make_zero_initializer(),
            },
        },
    };

    TrainingOperationAttrs linear_op_attrs = TrainingOperationAttrs{
        PCGOperatorAttrs{
            LinearAttrs{
                /*out_channels=*/6_p,
                /*use_bias=*/false,
                /*data_type=*/DataType::FLOAT,
                /*activation=*/std::nullopt,
                /*regularizer=*/std::nullopt,
            },
        },
    };

    DynamicOpenDataflowGraph input = [&]() -> DynamicOpenDataflowGraph {
      DynamicNodeAttrs input_node =
          mk_node_attrs(10, input_op_attrs, std::nullopt);
      DynamicNodeAttrs weight_node =
          mk_node_attrs(11, weight_op_attrs, std::nullopt);
      DynamicNodeAttrs relu_node =
          mk_node_attrs(12, relu_op_attrs, std::nullopt);
      DynamicNodeAttrs linear_node =
          mk_node_attrs(13, linear_op_attrs, std::nullopt);

      DynamicValueAttrs input_tensor = mk_value_attrs(0, std::nullopt);
      DynamicValueAttrs weight_tensor = mk_value_attrs(1, std::nullopt);
      DynamicValueAttrs relu_output = mk_value_attrs(2, std::nullopt);
      DynamicValueAttrs linear_output = mk_value_attrs(3, std::nullopt);

      auto mk_dynamic_slot =
          [](TensorSlotName const &slot_name) -> DynamicTensorSlot {
        return DynamicTensorSlot{
            /*slot_name=*/slot_name,
            /*slot_tensor_role=*/std::nullopt,
            /*task_shard=*/std::nullopt,
        };
      };

      std::set<DynamicNodeInvocation> invocation_set = {
          DynamicNodeInvocation{
              /*inputs=*/std::map<DynamicTensorSlot, DynamicValueAttrs>{},
              /*node_attrs=*/input_node,
              /*outputs=*/
              std::map<DynamicTensorSlot, DynamicValueAttrs>{
                  {
                      mk_dynamic_slot(TensorSlotName::OUTPUT),
                      input_tensor,
                  },
              },
          },
          DynamicNodeInvocation{
              /*inputs=*/std::map<DynamicTensorSlot, DynamicValueAttrs>{},
              /*node_attrs=*/weight_node,
              /*outputs=*/
              std::map<DynamicTensorSlot, DynamicValueAttrs>{
                  {
                      mk_dynamic_slot(TensorSlotName::OUTPUT),
                      weight_tensor,
                  },
              },
          },
          DynamicNodeInvocation{
              /*inputs=*/std::map<DynamicTensorSlot, DynamicValueAttrs>{
                  {
                      mk_dynamic_slot(TensorSlotName::INPUT),
                      input_tensor,
                  },
              },
              /*node_attrs=*/relu_node,
              /*outputs=*/
              std::map<DynamicTensorSlot, DynamicValueAttrs>{
                  {
                      mk_dynamic_slot(TensorSlotName::OUTPUT),
                      relu_output,
                  },
              },
          },
          DynamicNodeInvocation{
              /*inputs=*/std::map<DynamicTensorSlot, DynamicValueAttrs>{
                  {
                      mk_dynamic_slot(TensorSlotName::INPUT),
                      relu_output,
                  },
                  {
                      mk_dynamic_slot(TensorSlotName::WEIGHT),
                      weight_tensor,
                  },
              },
              /*node_attrs=*/linear_node,
              /*outputs=*/
              std::map<DynamicTensorSlot, DynamicValueAttrs>{
                  {
                      mk_dynamic_slot(TensorSlotName::OUTPUT),
                      linear_output,
                  },
              },
          },
      };

      return dynamic_open_dataflow_graph_from_invocation_set(invocation_set);
    }();

    DynamicOpenDataflowGraph result = perform_pass_expansion(input);

    DynamicOpenDataflowGraph correct = [&]() -> DynamicOpenDataflowGraph {
      DynamicNodeAttrs input_node_fwd =
          mk_node_attrs(10, input_op_attrs, DynamicTaskType::FWD);
      DynamicNodeAttrs weight_node_fwd =
          mk_node_attrs(11, weight_op_attrs, DynamicTaskType::FWD);
      DynamicNodeAttrs relu_node_fwd =
          mk_node_attrs(12, relu_op_attrs, DynamicTaskType::FWD);
      DynamicNodeAttrs linear_node_fwd =
          mk_node_attrs(13, linear_op_attrs, DynamicTaskType::FWD);

      DynamicNodeAttrs linear_node_bwd =
          mk_node_attrs(13, linear_op_attrs, DynamicTaskType::BWD);

      DynamicValueAttrs input_tensor_activation =
          mk_value_attrs(0, mk_dynamic_tensor_role_fwd());
      DynamicValueAttrs input_tensor_gradient =
          mk_value_attrs(0, mk_dynamic_tensor_role_bwd());
      DynamicValueAttrs weight_tensor_activation =
          mk_value_attrs(1, mk_dynamic_tensor_role_fwd());
      DynamicValueAttrs weight_tensor_gradient =
          mk_value_attrs(1, mk_dynamic_tensor_role_bwd());
      DynamicValueAttrs relu_output_tensor_activation =
          mk_value_attrs(2, mk_dynamic_tensor_role_fwd());
      DynamicValueAttrs relu_output_tensor_gradient =
          mk_value_attrs(2, mk_dynamic_tensor_role_bwd());
      DynamicValueAttrs linear_output_tensor_activation =
          mk_value_attrs(3, mk_dynamic_tensor_role_fwd());
      DynamicValueAttrs linear_output_tensor_gradient =
          mk_value_attrs(3, mk_dynamic_tensor_role_bwd());

      auto mk_fwd_slot = [&](TensorSlotName slot_name) -> DynamicTensorSlot {
        return DynamicTensorSlot{
            /*slot_name=*/slot_name,
            /*slot_tensor_role=*/mk_dynamic_tensor_role_fwd(),
            /*task_shard=*/std::nullopt,
        };
      };

      auto mk_grad_slot = [&](TensorSlotName slot_name) -> DynamicTensorSlot {
        return DynamicTensorSlot{
            /*slot_name=*/slot_name,
            /*slot_tensor_role=*/mk_dynamic_tensor_role_bwd(),
            /*task_shard=*/std::nullopt,
        };
      };

      std::set<DynamicNodeInvocation> invocation_set = {
          DynamicNodeInvocation{
              /*inputs=*/std::map<DynamicTensorSlot, DynamicValueAttrs>{},
              /*node_attrs=*/input_node_fwd,
              /*outputs=*/
              std::map{
                  std::pair{
                      mk_fwd_slot(TensorSlotName::OUTPUT),
                      input_tensor_activation,
                  },
              },
          },
          DynamicNodeInvocation{
              /*inputs=*/std::map<DynamicTensorSlot, DynamicValueAttrs>{},
              /*node_attrs=*/weight_node_fwd,
              /*outputs=*/
              std::map{
                  std::pair{
                      mk_fwd_slot(TensorSlotName::OUTPUT),
                      weight_tensor_activation,
                  },
              },
          },
          DynamicNodeInvocation{
              /*inputs=*/std::map{
                  std::pair{
                      mk_fwd_slot(TensorSlotName::INPUT),
                      input_tensor_activation,
                  },
              },
              /*node_attrs=*/relu_node_fwd,
              /*outputs=*/
              std::map{
                  std::pair{
                      mk_fwd_slot(TensorSlotName::OUTPUT),
                      relu_output_tensor_activation,
                  },
              },
          },
          DynamicNodeInvocation{
              /*inputs=*/std::map{
                  std::pair{
                      mk_fwd_slot(TensorSlotName::INPUT),
                      relu_output_tensor_activation,
                  },
                  std::pair{
                      mk_fwd_slot(TensorSlotName::WEIGHT),
                      weight_tensor_activation,
                  },
              },
              /*node_attrs=*/linear_node_fwd,
              /*outputs=*/
              std::map{
                  std::pair{
                      mk_fwd_slot(TensorSlotName::OUTPUT),
                      linear_output_tensor_activation,
                  },
              },
          },
          DynamicNodeInvocation{
              /*inputs=*/std::map{
                  std::pair{
                      mk_fwd_slot(TensorSlotName::INPUT),
                      relu_output_tensor_activation,
                  },
                  std::pair{
                      mk_fwd_slot(TensorSlotName::WEIGHT),
                      weight_tensor_activation,
                  },
                  std::pair{
                      mk_fwd_slot(TensorSlotName::OUTPUT),
                      linear_output_tensor_activation,
                  },
                  std::pair{
                      mk_grad_slot(TensorSlotName::OUTPUT),
                      linear_output_tensor_gradient,
                  },
              },
              /*node_attrs=*/linear_node_bwd,
              /*outputs=*/
              std::map{
                  std::pair{
                      mk_grad_slot(TensorSlotName::INPUT),
                      relu_output_tensor_gradient,
                  },
                  std::pair{
                      mk_grad_slot(TensorSlotName::WEIGHT),
                      weight_tensor_gradient,
                  },
              },
          },
      };

      return dynamic_open_dataflow_graph_from_invocation_set(invocation_set);
    }();

    CHECK(get_dynamic_invocation_set(result).size() ==
          correct.invocations.size());

    nlohmann::json result_json =
        dynamic_open_dataflow_graph_to_serializable(result);
    nlohmann::json correct_json =
        dynamic_open_dataflow_graph_to_serializable(correct);

    CHECK_MESSAGE(get_dynamic_invocation_set(result) ==
                      get_dynamic_invocation_set(correct),
                  check_kv("result", result_json.dump()),
                  check_kv("correct", correct_json.dump()));

    CHECK(dynamic_open_dataflow_graphs_are_isomorphic(result, correct));
  }

  TEST_CASE("perform_pass_expansion(DynamicOpenDataflowGraph) with multiple "
            "consumers") {
    auto mk_node_attrs = [](size_t layer_id,
                            TrainingOperationAttrs const &op_attrs,
                            std::optional<DynamicTaskType> const &pass_type)
        -> DynamicNodeAttrs {
      return DynamicNodeAttrs{
          /*pass_type=*/pass_type,
          /*device_coord=*/std::nullopt,
          /*mapping=*/std::nullopt,
          /*op_attrs=*/op_attrs,
          /*layer_guid=*/
          dynamic_layer_guid_t{parallel_layer_guid_t{Node{layer_id}}},
          /*per_device_op_state=*/std::nullopt,
      };
    };

    auto mk_gradient_node_attrs =
        [](std::optional<DynamicTaskType> const &pass_type)
        -> DynamicNodeAttrs {
      return DynamicNodeAttrs{
          /*pass_type=*/pass_type,
          /*device_coord=*/std::nullopt,
          /*mapping=*/std::nullopt,
          /*op_attrs=*/TrainingOperationAttrs{GradientReductionAttrs{}},
          /*layer_guid=*/
          dynamic_layer_guid_t{dynamic_gradient_reduction_layer_guid_t{}},
          /*per_device_op_state=*/std::nullopt,
      };
    };

    auto mk_value_attrs =
        [](size_t node_id,
           std::optional<DynamicTensorRole> const &tensor_type,
           std::optional<subgradient_id_t> const &subgradient_id)
        -> DynamicValueAttrs {
      return DynamicValueAttrs{
          /*tensor_guid=*/dynamic_tensor_guid_t{parallel_tensor_guid_t{
              KwargDataflowOutput{
                  Node{node_id},
                  TensorSlotName::OUTPUT,
              },
          }},
          /*parallel_tensor_shape=*/std::nullopt,
          /*create_grad=*/true,
          /*subgradient_id=*/subgradient_id,
          /*shard_coord=*/std::nullopt,
          /*mapping=*/std::nullopt,
          /*accessor=*/std::nullopt,
          /*role=*/tensor_type,
      };
    };

    TrainingOperationAttrs input_op_attrs = TrainingOperationAttrs{
        PCGOperatorAttrs{
            InputAttrs{
                TensorShape{
                    TensorDims{
                        FFOrdered<positive_int>{
                            4_p,
                            8_p,
                        },
                    },
                    DataType::FLOAT,
                },
            },
        },
    };

    TrainingOperationAttrs relu_op_attrs = TrainingOperationAttrs{
        PCGOperatorAttrs{
            make_relu_attrs(),
        },
    };

    DynamicOpenDataflowGraph input = [&]() -> DynamicOpenDataflowGraph {
      DynamicNodeAttrs input_node =
          mk_node_attrs(10, input_op_attrs, std::nullopt);
      DynamicNodeAttrs relu1_node =
          mk_node_attrs(11, relu_op_attrs, std::nullopt);
      DynamicNodeAttrs relu2_node =
          mk_node_attrs(12, relu_op_attrs, std::nullopt);

      DynamicValueAttrs input_tensor =
          mk_value_attrs(0, std::nullopt, std::nullopt);
      DynamicValueAttrs relu1_output =
          mk_value_attrs(1, std::nullopt, std::nullopt);
      DynamicValueAttrs relu2_output =
          mk_value_attrs(2, std::nullopt, std::nullopt);

      auto mk_dynamic_slot =
          [](TensorSlotName const &slot_name) -> DynamicTensorSlot {
        return DynamicTensorSlot{
            /*slot_name=*/slot_name,
            /*slot_tensor_role=*/std::nullopt,
            /*task_shard=*/std::nullopt,
        };
      };

      std::set<DynamicNodeInvocation> invocation_set = {
          DynamicNodeInvocation{
              /*inputs=*/std::map<DynamicTensorSlot, DynamicValueAttrs>{},
              /*node_attrs=*/input_node,
              /*outputs=*/
              std::map<DynamicTensorSlot, DynamicValueAttrs>{
                  {
                      mk_dynamic_slot(TensorSlotName::OUTPUT),
                      input_tensor,
                  },
              },
          },
          DynamicNodeInvocation{
              /*inputs=*/std::map<DynamicTensorSlot, DynamicValueAttrs>{
                  {
                      mk_dynamic_slot(TensorSlotName::INPUT),
                      input_tensor,
                  },
              },
              /*node_attrs=*/relu1_node,
              /*outputs=*/
              std::map<DynamicTensorSlot, DynamicValueAttrs>{
                  {
                      mk_dynamic_slot(TensorSlotName::OUTPUT),
                      relu1_output,
                  },
              },
          },
          DynamicNodeInvocation{
              /*inputs=*/std::map<DynamicTensorSlot, DynamicValueAttrs>{
                  {
                      mk_dynamic_slot(TensorSlotName::INPUT),
                      input_tensor,
                  },
              },
              /*node_attrs=*/relu2_node,
              /*outputs=*/
              std::map<DynamicTensorSlot, DynamicValueAttrs>{
                  {
                      mk_dynamic_slot(TensorSlotName::OUTPUT),
                      relu2_output,
                  },
              },
          },
      };

      return dynamic_open_dataflow_graph_from_invocation_set(invocation_set);
    }();

    DynamicOpenDataflowGraph result = perform_pass_expansion(input);

    DynamicOpenDataflowGraph correct = [&]() -> DynamicOpenDataflowGraph {
      DynamicNodeAttrs input_node_fwd =
          mk_node_attrs(10, input_op_attrs, DynamicTaskType::FWD);
      DynamicNodeAttrs relu1_node_fwd =
          mk_node_attrs(11, relu_op_attrs, DynamicTaskType::FWD);
      DynamicNodeAttrs relu2_node_fwd =
          mk_node_attrs(12, relu_op_attrs, DynamicTaskType::FWD);

      DynamicNodeAttrs relu1_node_bwd =
          mk_node_attrs(11, relu_op_attrs, DynamicTaskType::BWD);
      DynamicNodeAttrs relu2_node_bwd =
          mk_node_attrs(12, relu_op_attrs, DynamicTaskType::BWD);
      DynamicNodeAttrs gradient_reduction_node_bwd =
          mk_gradient_node_attrs(DynamicTaskType::BWD);

      DynamicValueAttrs input_tensor_activation =
          mk_value_attrs(0, mk_dynamic_tensor_role_fwd(), std::nullopt);
      DynamicValueAttrs input_tensor_subgradient1 =
          mk_value_attrs(0,
                         mk_dynamic_tensor_role_bwd(),
                         subgradient_id_t{TensorSlotName::INPUT_00});
      DynamicValueAttrs input_tensor_subgradient2 =
          mk_value_attrs(0,
                         mk_dynamic_tensor_role_bwd(),
                         subgradient_id_t{TensorSlotName::INPUT_01});
      DynamicValueAttrs input_tensor_gradient =
          mk_value_attrs(0, mk_dynamic_tensor_role_bwd(), std::nullopt);
      DynamicValueAttrs relu1_output_tensor_activation =
          mk_value_attrs(1, mk_dynamic_tensor_role_fwd(), std::nullopt);
      DynamicValueAttrs relu1_output_tensor_gradient =
          mk_value_attrs(1, mk_dynamic_tensor_role_bwd(), std::nullopt);
      DynamicValueAttrs relu2_output_tensor_activation =
          mk_value_attrs(2, mk_dynamic_tensor_role_fwd(), std::nullopt);
      DynamicValueAttrs relu2_output_tensor_gradient =
          mk_value_attrs(2, mk_dynamic_tensor_role_bwd(), std::nullopt);

      auto mk_fwd_slot = [&](TensorSlotName slot_name) -> DynamicTensorSlot {
        return DynamicTensorSlot{
            /*slot_name=*/slot_name,
            /*slot_tensor_role=*/mk_dynamic_tensor_role_fwd(),
            /*task_shard=*/std::nullopt,
        };
      };

      auto mk_grad_slot = [&](TensorSlotName slot_name) -> DynamicTensorSlot {
        return DynamicTensorSlot{
            /*slot_name=*/slot_name,
            /*slot_tensor_role=*/mk_dynamic_tensor_role_bwd(),
            /*task_shard=*/std::nullopt,
        };
      };

      std::set<DynamicNodeInvocation> invocation_set = {
          DynamicNodeInvocation{
              /*inputs=*/std::map<DynamicTensorSlot, DynamicValueAttrs>{},
              /*node_attrs=*/input_node_fwd,
              /*outputs=*/
              std::map{
                  std::pair{
                      mk_fwd_slot(TensorSlotName::OUTPUT),
                      input_tensor_activation,
                  },
              },
          },
          DynamicNodeInvocation{
              /*inputs=*/std::map{
                  std::pair{
                      mk_fwd_slot(TensorSlotName::INPUT),
                      input_tensor_activation,
                  },
              },
              /*node_attrs=*/relu1_node_fwd,
              /*outputs=*/
              std::map{
                  std::pair{
                      mk_fwd_slot(TensorSlotName::OUTPUT),
                      relu1_output_tensor_activation,
                  },
              },
          },
          DynamicNodeInvocation{
              /*inputs=*/std::map{
                  std::pair{
                      mk_fwd_slot(TensorSlotName::INPUT),
                      input_tensor_activation,
                  },
              },
              /*node_attrs=*/relu2_node_fwd,
              /*outputs=*/
              std::map{
                  std::pair{
                      mk_fwd_slot(TensorSlotName::OUTPUT),
                      relu2_output_tensor_activation,
                  },
              },
          },
          DynamicNodeInvocation{
              /*inputs=*/std::map{
                  std::pair{
                      mk_fwd_slot(TensorSlotName::INPUT),
                      input_tensor_activation,
                  },
                  std::pair{
                      mk_fwd_slot(TensorSlotName::OUTPUT),
                      relu1_output_tensor_activation,
                  },
                  std::pair{
                      mk_grad_slot(TensorSlotName::OUTPUT),
                      relu1_output_tensor_gradient,
                  },
              },
              /*node_attrs=*/relu1_node_bwd,
              /*outputs=*/
              std::map{
                  std::pair{
                      mk_grad_slot(TensorSlotName::INPUT),
                      input_tensor_subgradient1,
                  },
              },
          },
          DynamicNodeInvocation{
              /*inputs=*/std::map{
                  std::pair{
                      mk_fwd_slot(TensorSlotName::INPUT),
                      input_tensor_activation,
                  },
                  std::pair{
                      mk_fwd_slot(TensorSlotName::OUTPUT),
                      relu2_output_tensor_activation,
                  },
                  std::pair{
                      mk_grad_slot(TensorSlotName::OUTPUT),
                      relu2_output_tensor_gradient,
                  },
              },
              /*node_attrs=*/relu2_node_bwd,
              /*outputs=*/
              std::map{
                  std::pair{
                      mk_grad_slot(TensorSlotName::INPUT),
                      input_tensor_subgradient2,
                  },
              },
          },
          DynamicNodeInvocation{
              /*inputs=*/std::map{
                  std::pair{
                      mk_grad_slot(TensorSlotName::INPUT_00),
                      input_tensor_subgradient1,
                  },
                  std::pair{
                      mk_grad_slot(TensorSlotName::INPUT_01),
                      input_tensor_subgradient2,
                  },
              },
              /*node_attrs=*/gradient_reduction_node_bwd,
              /*outputs=*/
              std::map{
                  std::pair{
                      mk_grad_slot(TensorSlotName::OUTPUT),
                      input_tensor_gradient,
                  },
              },
          },
      };

      return dynamic_open_dataflow_graph_from_invocation_set(invocation_set);
    }();

    CHECK(get_dynamic_invocation_set(result).size() ==
          correct.invocations.size());

    nlohmann::json result_json =
        dynamic_open_dataflow_graph_to_serializable(result);
    nlohmann::json correct_json =
        dynamic_open_dataflow_graph_to_serializable(correct);

    CHECK_MESSAGE(get_dynamic_invocation_set(result) ==
                      get_dynamic_invocation_set(correct),
                  check_kv("result", result_json.dump()),
                  check_kv("correct", correct_json.dump()));

    CHECK(dynamic_open_dataflow_graphs_are_isomorphic(result, correct));
  }
}
