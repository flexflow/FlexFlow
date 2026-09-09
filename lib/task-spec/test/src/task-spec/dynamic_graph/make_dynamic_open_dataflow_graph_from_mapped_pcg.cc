#include "task-spec/dynamic_graph/make_dynamic_open_dataflow_graph_from_mapped_pcg.h"
#include "op-attrs/initializer_attrs.h"
#include "op-attrs/ops/element_unary.h"
#include "pcg/mapped_parallel_computation_graph/mapped_parallel_computation_graph.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/serializable_dynamic_node_invocation.h"
#include "task-spec/dynamic_graph/serializable_dynamic_open_dataflow_graph.h"
#include "test/utils/doctest/check_kv.h"
#include "utils/containers/require_only_key.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("make_dynamic_node_invocation_from_mapped") {
    MachineSpaceCoordinate gpu0 = MachineSpaceCoordinate{0_n, 0_n};
    MachineSpaceCoordinate gpu1 = MachineSpaceCoordinate{0_n, 1_n};

    SUBCASE("Replicate") {
      ParallelTensorSpaceCoordinate tensor_coord0 =
          ParallelTensorSpaceCoordinate{
              /*sum_component=*/0_n,
              /*discard_copy_component=*/0_n,
              /*shard_component=*/FFOrdered{0_n},
          };

      ParallelTensorSpaceCoordinate tensor_coord1 =
          ParallelTensorSpaceCoordinate{
              /*sum_component=*/0_n,
              /*discard_copy_component=*/1_n,
              /*shard_component=*/FFOrdered{0_n},
          };

      MappedOperatorTaskGroup mapped_op_task_group = MappedOperatorTaskGroup{
          {
              {
                  gpu0,
                  OperatorAtomicTaskShardBinding{{
                      {TensorSlotName::OUTPUT, tensor_coord0},
                  }},
              },
              {
                  gpu1,
                  OperatorAtomicTaskShardBinding{{
                      {TensorSlotName::OUTPUT, tensor_coord1},
                  }},
              },
          },
      };

      ParallelTensorShape input_shape = ParallelTensorShape{
          /*dims=*/ParallelTensorDims{
              /*shard_dims=*/FFOrdered<ShardParallelDim>{
                  ShardParallelDim{8_p, 2_p},
                  ShardParallelDim{5_p, 1_p},
              },
              /*replica_dims=*/
              ReplicaParallelDimSet{
                  SumDegree{1_p},
                  DiscardCopyDegree{1_p},
              },
          },
          /*data_type=*/DataType::FLOAT,
      };

      ParallelTensorShape output_shape = [&] {
        ParallelTensorShape shape = input_shape;
        shape.dims.replica_dims.discard_copy_degree = DiscardCopyDegree{2_p};
        return shape;
      }();

      PCGOperatorAttrs op_attrs = PCGOperatorAttrs{
          ReplicateAttrs{
              2_p,
          },
      };

      parallel_layer_guid_t layer_guid = parallel_layer_guid_t{Node{0}};
      parallel_tensor_guid_t input_tensor_guid = parallel_tensor_guid_t{
          KwargDataflowOutput{
              Node{5},
              TensorSlotName::OUTPUT,
          },
      };
      parallel_tensor_guid_t output_tensor_guid = parallel_tensor_guid_t{
          KwargDataflowOutput{
              Node{0},
              TensorSlotName::OUTPUT,
          },
      };

      MappedParallelLayerInvocationInfo input =
          MappedParallelLayerInvocationInfo{
              /*incoming=*/{
                  {
                      TensorSlotName::INPUT,
                      ParallelTensorInfo{
                          /*guid=*/input_tensor_guid,
                          /*attrs=*/
                          ParallelTensorAttrs{
                              /*shape=*/input_shape,
                              /*create_grad=*/CreateGrad::YES,
                          },
                      },
                  },
              },
              /*layer_info=*/
              MappedParallelLayerInfo{
                  /*guid=*/layer_guid,
                  /*attrs=*/
                  ParallelLayerAttrs{
                      /*op_attrs=*/op_attrs,
                      /*name=*/std::nullopt,
                  },
                  /*mapping=*/mapped_op_task_group,
              },
              /*outgoing=*/
              {
                  {
                      TensorSlotName::OUTPUT,
                      ParallelTensorInfo{
                          /*guid=*/output_tensor_guid,
                          /*attrs=*/
                          ParallelTensorAttrs{
                              /*shape=*/output_shape,
                              /*create_grad=*/CreateGrad::YES,
                          },
                      },
                  },
              },
          };

      DynamicNodeInvocation result =
          make_dynamic_node_invocation_from_mapped(input, DeviceType::GPU);

      DynamicNodeInvocation correct = DynamicNodeInvocation{
          /*inputs=*/{
              {
                  DynamicTensorSlot{
                      TensorSlotName::INPUT,
                      /*slot_tensor_role=*/std::nullopt,
                      /*task_shard=*/std::nullopt,
                  },
                  DynamicValueAttrs{
                      /*tensor_guid=*/dynamic_tensor_guid_t{input_tensor_guid},
                      /*parallel_tensor_shape=*/input_shape,
                      /*create_grad=*/true,
                      /*subgradient_id=*/std::nullopt,
                      /*shard_coord=*/std::nullopt,
                      /*mapping=*/std::nullopt,
                      /*accessor=*/std::nullopt,
                      /*role=*/std::nullopt,
                  },
              },
          },
          /*node_attrs=*/
          DynamicNodeAttrs{
              /*task_type=*/std::nullopt,
              /*device_coord=*/std::nullopt,
              /*mapping=*/
              DynamicNodeMapping{
                  /*op_task_group=*/mapped_op_task_group,
                  /*device_type=*/DeviceType::GPU,
              },
              /*op_attrs=*/TrainingOperationAttrs{op_attrs},
              /*layer_guid=*/dynamic_layer_guid_t{layer_guid},
              /*per_device_op_state=*/std::nullopt,
          },
          /*outputs=*/
          {
              {
                  DynamicTensorSlot{
                      TensorSlotName::OUTPUT,
                      /*slot_tensor_role=*/std::nullopt,
                      /*task_shard=*/std::nullopt,
                  },
                  DynamicValueAttrs{
                      /*tensor_guid=*/dynamic_tensor_guid_t{output_tensor_guid},
                      /*parallel_tensor_shape=*/output_shape,
                      /*create_grad=*/true,
                      /*subgradient_id=*/std::nullopt,
                      /*shard_coord=*/std::nullopt,
                      /*mapping=*/std::nullopt,
                      /*accessor=*/std::nullopt,
                      /*role=*/std::nullopt,
                  },
              },
          }};

      nlohmann::json result_json =
          dynamic_node_invocation_to_serializable(result);
      nlohmann::json correct_json =
          dynamic_node_invocation_to_serializable(correct);

      CHECK_MESSAGE(result == correct,
                    check_kv("result\n", result_json.dump()),
                    check_kv("correct\n", correct_json.dump()));
    }

    SUBCASE("standard op") {
      PCGOperatorAttrs op_attrs = PCGOperatorAttrs{
          LinearAttrs{
              /*out_channels=*/7_p,
              /*use_bias=*/true,
              /*data_type=*/DataType::FLOAT,
              /*activation=*/std::nullopt,
              /*regularizer=*/std::nullopt,
          },
      };

      parallel_layer_guid_t layer_guid = parallel_layer_guid_t{Node{0}};

      auto mk_tensor_guid = [](size_t node_id) -> parallel_tensor_guid_t {
        return parallel_tensor_guid_t{
            KwargDataflowOutput{
                Node{node_id},
                TensorSlotName::OUTPUT,
            },
        };
      };

      parallel_tensor_guid_t input_tensor_guid = mk_tensor_guid(5);
      parallel_tensor_guid_t weight_tensor_guid = mk_tensor_guid(6);
      parallel_tensor_guid_t bias_tensor_guid = mk_tensor_guid(7);
      parallel_tensor_guid_t output_tensor_guid = mk_tensor_guid(0);

      ParallelTensorShape input_shape = ParallelTensorShape{
          /*dims=*/ParallelTensorDims{
              /*shard_dims=*/FFOrdered<ShardParallelDim>{
                  ShardParallelDim{8_p, 2_p},
                  ShardParallelDim{5_p, 1_p},
              },
              /*replica_dims=*/
              ReplicaParallelDimSet{
                  SumDegree{1_p},
                  DiscardCopyDegree{1_p},
              },
          },
          /*data_type=*/DataType::FLOAT,
      };

      ParallelTensorShape weight_shape = ParallelTensorShape{
          /*dims=*/ParallelTensorDims{
              /*shard_dims=*/FFOrdered<ShardParallelDim>{
                  ShardParallelDim{5_p, 1_p},
                  ShardParallelDim{7_p, 1_p},
              },
              /*replica_dims=*/
              ReplicaParallelDimSet{
                  SumDegree{1_p},
                  DiscardCopyDegree{2_p},
              },
          },
          /*data_type=*/DataType::FLOAT,
      };

      ParallelTensorShape bias_shape = ParallelTensorShape{
          /*dims=*/ParallelTensorDims{
              /*shard_dims=*/FFOrdered<ShardParallelDim>{
                  ShardParallelDim{7_p, 1_p},
              },
              /*replica_dims=*/
              ReplicaParallelDimSet{
                  SumDegree{1_p},
                  DiscardCopyDegree{2_p},
              },
          },
          /*data_type=*/DataType::FLOAT,
      };

      ParallelTensorShape output_shape = ParallelTensorShape{
          /*dims=*/ParallelTensorDims{
              /*shard_dims=*/FFOrdered<ShardParallelDim>{
                  ShardParallelDim{8_p, 2_p},
                  ShardParallelDim{7_p, 1_p},
              },
              /*replica_dims=*/
              ReplicaParallelDimSet{
                  SumDegree{1_p},
                  DiscardCopyDegree{1_p},
              },
          },
          /*data_type=*/DataType::FLOAT,
      };

      auto mk_2d_pt_coord =
          [](nonnegative_int replica_coord,
             nonnegative_int shard_coord) -> ParallelTensorSpaceCoordinate {
        return ParallelTensorSpaceCoordinate{
            /*sum_component=*/0_n,
            /*discard_copy_compnent=*/replica_coord,
            /*shard_components=*/
            FFOrdered<nonnegative_int>{
                shard_coord,
                0_n,
            },
        };
      };

      auto mk_1d_pt_coord =
          [](nonnegative_int replica_coord,
             nonnegative_int shard_coord) -> ParallelTensorSpaceCoordinate {
        return ParallelTensorSpaceCoordinate{
            /*sum_component=*/0_n,
            /*discard_copy_compnent=*/replica_coord,
            /*shard_components=*/
            FFOrdered<nonnegative_int>{
                shard_coord,
            },
        };
      };

      MappedOperatorTaskGroup mapped_op_task_group = MappedOperatorTaskGroup{
          {
              {
                  gpu0,
                  OperatorAtomicTaskShardBinding{{
                      {TensorSlotName::INPUT, mk_2d_pt_coord(0_n, 0_n)},
                      {TensorSlotName::WEIGHT, mk_2d_pt_coord(0_n, 0_n)},
                      {TensorSlotName::BIAS, mk_1d_pt_coord(0_n, 0_n)},
                      {TensorSlotName::OUTPUT, mk_2d_pt_coord(0_n, 0_n)},
                  }},
              },
              {
                  gpu1,
                  OperatorAtomicTaskShardBinding{{
                      {TensorSlotName::INPUT, mk_2d_pt_coord(0_n, 1_n)},
                      {TensorSlotName::WEIGHT, mk_2d_pt_coord(1_n, 0_n)},
                      {TensorSlotName::BIAS, mk_1d_pt_coord(1_n, 0_n)},
                      {TensorSlotName::OUTPUT, mk_2d_pt_coord(0_n, 1_n)},
                  }},
              },
          },
      };

      MappedParallelLayerInvocationInfo input =
          MappedParallelLayerInvocationInfo{
              /*incoming=*/{
                  {
                      TensorSlotName::INPUT,
                      ParallelTensorInfo{
                          /*guid=*/input_tensor_guid,
                          /*attrs=*/
                          ParallelTensorAttrs{
                              /*shape=*/input_shape,
                              /*create_grad=*/CreateGrad::YES,
                          },
                      },
                  },
                  {
                      TensorSlotName::WEIGHT,
                      ParallelTensorInfo{
                          /*guid=*/weight_tensor_guid,
                          /*attrs=*/
                          ParallelTensorAttrs{
                              /*shape=*/weight_shape,
                              /*create_grad=*/CreateGrad::YES,
                          },
                      },
                  },
                  {
                      TensorSlotName::BIAS,
                      ParallelTensorInfo{
                          /*guid=*/bias_tensor_guid,
                          /*attrs=*/
                          ParallelTensorAttrs{
                              /*shape=*/bias_shape,
                              /*create_grad=*/CreateGrad::YES,
                          },
                      },
                  },
              },
              /*layer_info=*/
              MappedParallelLayerInfo{
                  /*guid=*/layer_guid,
                  /*attrs=*/
                  ParallelLayerAttrs{
                      /*op_attrs=*/op_attrs,
                      /*name=*/std::nullopt,
                  },
                  /*mapping=*/mapped_op_task_group,
              },
              /*outgoing=*/
              {
                  {
                      TensorSlotName::OUTPUT,
                      ParallelTensorInfo{
                          /*guid=*/output_tensor_guid,
                          /*attrs=*/
                          ParallelTensorAttrs{
                              /*shape=*/output_shape,
                              /*create_grad=*/CreateGrad::YES,
                          },
                      },
                  },
              },
          };

      DynamicNodeInvocation result =
          make_dynamic_node_invocation_from_mapped(input, DeviceType::GPU);

      DynamicNodeInvocation correct = [&]() -> DynamicNodeInvocation {
        auto mk_slot = [](TensorSlotName slot_name) -> DynamicTensorSlot {
          return DynamicTensorSlot{
              /*slot_name=*/slot_name,
              /*slot_tensor_role=*/std::nullopt,
              /*task_shard=*/std::nullopt,
          };
        };

        auto mk_value =
            [](parallel_tensor_guid_t const &tensor_guid,
               ParallelTensorShape const &shape) -> DynamicValueAttrs {
          return DynamicValueAttrs{
              /*tensor_guid=*/dynamic_tensor_guid_t{tensor_guid},
              /*parallel_tensor_shape=*/shape,
              /*create_grad=*/true,
              /*subgradient_id=*/std::nullopt,
              /*shard_coord=*/std::nullopt,
              /*mapping=*/std::nullopt,
              /*accessor=*/std::nullopt,
              /*role=*/std::nullopt,
          };
        };

        return DynamicNodeInvocation{
            /*inputs=*/{
                {
                    mk_slot(TensorSlotName::INPUT),
                    mk_value(input_tensor_guid, input_shape),
                },
                {
                    mk_slot(TensorSlotName::WEIGHT),
                    mk_value(weight_tensor_guid, weight_shape),
                },
                {
                    mk_slot(TensorSlotName::BIAS),
                    mk_value(bias_tensor_guid, bias_shape),
                },
            },
            /*node_attrs=*/
            DynamicNodeAttrs{
                /*task_type=*/std::nullopt,
                /*device_coord=*/std::nullopt,
                /*mapping=*/
                DynamicNodeMapping{
                    /*op_task_group=*/mapped_op_task_group,
                    /*device_type=*/DeviceType::GPU,
                },
                /*op_attrs=*/TrainingOperationAttrs{op_attrs},
                /*layer_guid=*/dynamic_layer_guid_t{layer_guid},
                /*per_device_op_state=*/std::nullopt,
            },
            /*outputs=*/
            {
                {
                    mk_slot(TensorSlotName::OUTPUT),
                    mk_value(output_tensor_guid, output_shape),
                },
            }};
      }();

      nlohmann::json result_json =
          dynamic_node_invocation_to_serializable(result);
      nlohmann::json correct_json =
          dynamic_node_invocation_to_serializable(correct);

      CHECK_MESSAGE(result == correct,
                    check_kv("result\n", result_json.dump()),
                    check_kv("correct\n", correct_json.dump()));
    }
  }

  TEST_CASE("make_dynamic_open_dataflow_graph_from_mapped_pcg") {
    positive_int batch_size = 10_p;
    positive_int data_dim = 16_p;
    positive_int hidden_dim = 32_p;
    positive_int output_dim = 1_p;

    auto make_layer_attrs =
        [](PCGOperatorAttrs const &op_attrs) -> ParallelLayerAttrs {
      return ParallelLayerAttrs{
          /*op_attrs=*/op_attrs,
          /*name=*/std::nullopt,
      };
    };

    TensorShape input_tensor_shape = TensorShape{
        TensorDims{FFOrdered{batch_size, data_dim}}, DataType::FLOAT};

    TensorShape weight_1_tensor_shape = TensorShape{
        TensorDims{FFOrdered{hidden_dim, data_dim}}, DataType::FLOAT};

    TensorShape weight_2_tensor_shape = TensorShape{
        TensorDims{FFOrdered{output_dim, hidden_dim}}, DataType::FLOAT};

    TensorShape output_tensor_shape = TensorShape{
        TensorDims{FFOrdered{batch_size, output_dim}}, DataType::FLOAT};

    TensorShape label_tensor_shape = TensorShape{
        TensorDims{FFOrdered{batch_size, output_dim}}, DataType::FLOAT};

    ParallelComputationGraph pcg = empty_parallel_computation_graph();

    PCGOperatorAttrs input_op_attrs = PCGOperatorAttrs{
        InputAttrs{/*tensor_shape=*/input_tensor_shape},
    };

    PCGOperatorAttrs partition_input_op_attrs = PCGOperatorAttrs{
        RepartitionAttrs{
            /*repartition_dim=*/ff_dim_t{0_n},
            /*repartition_degree=*/2_p,
        },
    };

    PCGOperatorAttrs weight_1_op_attrs = PCGOperatorAttrs{
        WeightAttrs{
            /*tensor_shape=*/weight_1_tensor_shape,
            /*initializer=*/make_kaiming_uniform(weight_1_tensor_shape.dims),
        },
    };

    PCGOperatorAttrs replicate_weight_1_op_attrs = PCGOperatorAttrs{
        ReplicateAttrs{
            /*replicate_degree=*/2_p,
        },
    };

    PCGOperatorAttrs weight_2_op_attrs = PCGOperatorAttrs{
        WeightAttrs{
            /*tensor_shape=*/weight_2_tensor_shape,
            /*initializer=*/make_kaiming_uniform(weight_1_tensor_shape.dims),
        },
    };

    PCGOperatorAttrs replicate_weight_2_op_attrs = PCGOperatorAttrs{
        ReplicateAttrs{
            /*replicate_degree=*/2_p,
        },
    };

    PCGOperatorAttrs linear_1_op_attrs = PCGOperatorAttrs{
        LinearAttrs{
            /*out_channels=*/hidden_dim,
            /*use_bias=*/false,
            /*data_type=*/DataType::FLOAT,
            /*activation=*/std::nullopt,
            /*regularizer=*/std::nullopt,
        },
    };

    PCGOperatorAttrs linear_2_op_attrs = PCGOperatorAttrs{
        LinearAttrs{
            /*out_channels=*/output_dim,
            /*use_bias=*/false,
            /*data_type=*/DataType::FLOAT,
            /*activation=*/std::nullopt,
            /*regularizer=*/std::nullopt,
        },
    };

    ParallelLayerAddedResult input_layer =
        pcg_add_input_layer(pcg, input_tensor_shape);
    parallel_tensor_guid_t t_input =
        require_only_key(input_layer.outputs, TensorSlotName::OUTPUT);

    ParallelLayerAddedResult partition_input_layer =
        add_parallel_layer(pcg,
                           make_layer_attrs(partition_input_op_attrs),
                           /*inputs=*/
                           {
                               {TensorSlotName::INPUT, t_input},
                           },
                           /*weights=*/{});
    parallel_tensor_guid_t t_partitioned_input =
        require_only_key(partition_input_layer.outputs, TensorSlotName::OUTPUT);

    ParallelLayerAddedResult weight_1_layer =
        add_parallel_layer(pcg,
                           make_layer_attrs(weight_1_op_attrs),
                           /*inputs=*/{},
                           /*weights=*/{});
    parallel_tensor_guid_t t_weight_1 =
        require_only_key(weight_1_layer.outputs, TensorSlotName::OUTPUT);

    ParallelLayerAddedResult replicate_weight_1_layer =
        add_parallel_layer(pcg,
                           make_layer_attrs(replicate_weight_1_op_attrs),
                           /*inputs=*/
                           {
                               {TensorSlotName::INPUT, t_weight_1},
                           },
                           /*weights=*/{});
    parallel_tensor_guid_t t_replicated_weight_1 = require_only_key(
        replicate_weight_1_layer.outputs, TensorSlotName::OUTPUT);

    ParallelLayerAddedResult weight_2_layer =
        add_parallel_layer(pcg,
                           make_layer_attrs(weight_2_op_attrs),
                           /*inputs=*/{},
                           /*weights=*/{});
    parallel_tensor_guid_t t_weight_2 =
        require_only_key(weight_2_layer.outputs, TensorSlotName::OUTPUT);

    ParallelLayerAddedResult replicate_weight_2_layer =
        add_parallel_layer(pcg,
                           make_layer_attrs(replicate_weight_2_op_attrs),
                           /*inputs=*/
                           {
                               {TensorSlotName::INPUT, t_weight_2},
                           },
                           /*weights=*/{});
    parallel_tensor_guid_t t_replicated_weight_2 = require_only_key(
        replicate_weight_2_layer.outputs, TensorSlotName::OUTPUT);

    ParallelLayerAddedResult linear_1_layer =
        add_parallel_layer(pcg,
                           make_layer_attrs(linear_1_op_attrs),
                           /*inputs=*/
                           {
                               {TensorSlotName::INPUT, t_partitioned_input},
                           },
                           /*weights=*/
                           {
                               {TensorSlotName::WEIGHT, t_replicated_weight_1},
                           });
    parallel_tensor_guid_t t_hidden_activation =
        require_only_key(linear_1_layer.outputs, TensorSlotName::OUTPUT);

    ParallelLayerAddedResult linear_2_layer =
        add_parallel_layer(pcg,
                           make_layer_attrs(linear_2_op_attrs),
                           /*inputs=*/
                           {
                               {TensorSlotName::INPUT, t_hidden_activation},
                           },
                           /*weights=*/
                           {
                               {TensorSlotName::WEIGHT, t_replicated_weight_2},
                           });
    parallel_tensor_guid_t t_output =
        require_only_key(linear_2_layer.outputs, TensorSlotName::OUTPUT);

    MachineSpaceCoordinate gpu0 = MachineSpaceCoordinate{
        /*node_idx=*/0_n,
        /*device_idx=*/0_n,
    };
    MachineSpaceCoordinate gpu1 = MachineSpaceCoordinate{
        /*node_idx=*/0_n,
        /*device_idx=*/1_n,
    };

    auto mk_shard_coord =
        [](nonnegative_int shard_idx) -> ParallelTensorSpaceCoordinate {
      return ParallelTensorSpaceCoordinate{
          /*sum_component=*/0_n,
          /*discard_copy_component=*/0_n,
          /*shard_component=*/FFOrdered{shard_idx, 0_n},
      };
    };

    auto mk_replica_coord =
        [](nonnegative_int replica_idx) -> ParallelTensorSpaceCoordinate {
      return ParallelTensorSpaceCoordinate{
          /*sum_component=*/0_n,
          /*discard_copy_component=*/replica_idx,
          /*shard_component=*/FFOrdered{0_n, 0_n},
      };
    };

    ParallelTensorSpaceCoordinate nonparallel_coord =
        ParallelTensorSpaceCoordinate{
            /*sum_component=*/0_n,
            /*discard_copy_component=*/0_n,
            /*shard_component=*/FFOrdered{0_n, 0_n},
        };

    MappedOperatorTaskGroup input_op_mapping = MappedOperatorTaskGroup{
        {
            {
                gpu0,
                OperatorAtomicTaskShardBinding{{
                    {TensorSlotName::OUTPUT, nonparallel_coord},
                }},
            },
        },
    };

    MappedOperatorTaskGroup weight_1_op_mapping = MappedOperatorTaskGroup{
        {
            {
                gpu0,
                OperatorAtomicTaskShardBinding{{
                    {TensorSlotName::OUTPUT, nonparallel_coord},
                }},
            },
        },
    };

    MappedOperatorTaskGroup weight_2_op_mapping = MappedOperatorTaskGroup{
        {
            {
                gpu0,
                OperatorAtomicTaskShardBinding{{
                    {TensorSlotName::OUTPUT, nonparallel_coord},
                }},
            },
        },
    };

    MappedOperatorTaskGroup partition_input_op_mapping =
        MappedOperatorTaskGroup{
            {
                {
                    gpu0,
                    OperatorAtomicTaskShardBinding{{
                        {TensorSlotName::INPUT, nonparallel_coord},
                        {TensorSlotName::OUTPUT, mk_shard_coord(0_n)},
                    }},
                },
                {
                    gpu1,
                    OperatorAtomicTaskShardBinding{{
                        {TensorSlotName::INPUT, nonparallel_coord},
                        {TensorSlotName::OUTPUT, mk_shard_coord(1_n)},
                    }},
                },
            },
        };

    MappedOperatorTaskGroup replicate_weight_1_op_mapping =
        MappedOperatorTaskGroup{
            {
                {
                    gpu0,
                    OperatorAtomicTaskShardBinding{{
                        {TensorSlotName::INPUT, nonparallel_coord},
                        {TensorSlotName::OUTPUT, mk_replica_coord(0_n)},
                    }},
                },
                {
                    gpu1,
                    OperatorAtomicTaskShardBinding{{
                        {TensorSlotName::INPUT, nonparallel_coord},
                        {TensorSlotName::OUTPUT, mk_replica_coord(1_n)},
                    }},
                },
            },
        };

    MappedOperatorTaskGroup replicate_weight_2_op_mapping =
        MappedOperatorTaskGroup{
            {
                {
                    gpu0,
                    OperatorAtomicTaskShardBinding{{
                        {TensorSlotName::INPUT, nonparallel_coord},
                        {TensorSlotName::OUTPUT, mk_replica_coord(0_n)},
                    }},
                },
                {
                    gpu1,
                    OperatorAtomicTaskShardBinding{{
                        {TensorSlotName::INPUT, nonparallel_coord},
                        {TensorSlotName::OUTPUT, mk_replica_coord(1_n)},
                    }},
                },
            },
        };

    MappedOperatorTaskGroup linear_1_op_mapping = MappedOperatorTaskGroup{
        {
            {
                gpu0,
                OperatorAtomicTaskShardBinding{{
                    {TensorSlotName::INPUT, mk_shard_coord(0_n)},
                    {TensorSlotName::WEIGHT, mk_replica_coord(0_n)},
                    {TensorSlotName::OUTPUT, mk_shard_coord(0_n)},
                }},
            },
            {
                gpu1,
                OperatorAtomicTaskShardBinding{{
                    {TensorSlotName::INPUT, mk_shard_coord(1_n)},
                    {TensorSlotName::WEIGHT, mk_replica_coord(1_n)},
                    {TensorSlotName::OUTPUT, mk_shard_coord(1_n)},
                }},
            },
        },
    };

    MappedOperatorTaskGroup linear_2_op_mapping = MappedOperatorTaskGroup{
        {
            {
                gpu0,
                OperatorAtomicTaskShardBinding{{
                    {TensorSlotName::INPUT, mk_shard_coord(0_n)},
                    {TensorSlotName::WEIGHT, mk_replica_coord(0_n)},
                    {TensorSlotName::OUTPUT, mk_shard_coord(0_n)},
                }},
            },
            {
                gpu1,
                OperatorAtomicTaskShardBinding{{
                    {TensorSlotName::INPUT, mk_shard_coord(1_n)},
                    {TensorSlotName::WEIGHT, mk_replica_coord(1_n)},
                    {TensorSlotName::OUTPUT, mk_shard_coord(1_n)},
                }},
            },
        },
    };

    MappedParallelComputationGraph mpcg =
        mapped_pcg_from_pcg_and_mapped_op_task_groups(
            /*pcg=*/pcg,
            /*mapped_op_task_groups=*/{
                {
                    input_layer.parallel_layer,
                    input_op_mapping,
                },
                {
                    weight_1_layer.parallel_layer,
                    weight_1_op_mapping,
                },
                {
                    weight_2_layer.parallel_layer,
                    weight_2_op_mapping,
                },
                {
                    partition_input_layer.parallel_layer,
                    partition_input_op_mapping,
                },
                {
                    replicate_weight_1_layer.parallel_layer,
                    replicate_weight_1_op_mapping,
                },
                {
                    replicate_weight_2_layer.parallel_layer,
                    replicate_weight_2_op_mapping,
                },
                {
                    linear_1_layer.parallel_layer,
                    linear_1_op_mapping,
                },
                {
                    linear_2_layer.parallel_layer,
                    linear_2_op_mapping,
                },
            });

    DynamicOpenDataflowGraph result =
        make_dynamic_open_dataflow_graph_from_mapped_pcg(mpcg, DeviceType::GPU);

    DynamicOpenDataflowGraph correct = [&]() -> DynamicOpenDataflowGraph {
      auto mk_pt_shape =
          [](positive_int sum_degree,
             positive_int discard_copy_degree,
             positive_int shard_dim_size_0,
             positive_int shard_degree_0,
             positive_int shard_dim_size_1,
             positive_int shard_degree_1) -> ParallelTensorShape {
        return ParallelTensorShape{
            /*dims=*/ParallelTensorDims{
                /*shard_dims=*/FFOrdered{
                    ShardParallelDim{shard_dim_size_0, shard_degree_0},
                    ShardParallelDim{shard_dim_size_1, shard_degree_1},
                },
                /*replica_dims=*/
                ReplicaParallelDimSet{
                    /*sum_degree=*/SumDegree{sum_degree},
                    /*discard_copy_degree=*/
                    DiscardCopyDegree{discard_copy_degree},
                },
            },
            /*data_type=*/DataType::FLOAT,
        };
      };

      ParallelTensorShape input_pt_shape =
          mk_pt_shape(1_p, 1_p, batch_size, 1_p, data_dim, 1_p);
      ParallelTensorShape weight_1_pt_shape =
          mk_pt_shape(1_p, 1_p, hidden_dim, 1_p, data_dim, 1_p);
      ParallelTensorShape weight_2_pt_shape =
          mk_pt_shape(1_p, 1_p, output_dim, 1_p, hidden_dim, 1_p);
      ParallelTensorShape partitioned_input_pt_shape =
          mk_pt_shape(1_p, 1_p, batch_size, 2_p, data_dim, 1_p);
      ParallelTensorShape replicated_weight_1_pt_shape =
          mk_pt_shape(1_p, 2_p, hidden_dim, 1_p, data_dim, 1_p);
      ParallelTensorShape replicated_weight_2_pt_shape =
          mk_pt_shape(1_p, 2_p, output_dim, 1_p, hidden_dim, 1_p);
      ParallelTensorShape hidden_activation_pt_shape =
          mk_pt_shape(1_p, 1_p, batch_size, 2_p, hidden_dim, 1_p);
      ParallelTensorShape output_pt_shape =
          mk_pt_shape(1_p, 1_p, batch_size, 2_p, output_dim, 1_p);

      auto mk_node_attrs =
          [&](parallel_layer_guid_t const &layer_guid,
              PCGOperatorAttrs const &op_attrs,
              MappedOperatorTaskGroup const &mapped_op_task_group)
          -> DynamicNodeAttrs {
        return DynamicNodeAttrs{
            /*task_type=*/std::nullopt,
            /*device_coord=*/std::nullopt,
            /*mapping=*/
            DynamicNodeMapping{
                mapped_op_task_group,
                DeviceType::GPU,
            },
            /*op_attrs=*/TrainingOperationAttrs{op_attrs},
            /*pcg_layer_guid=*/dynamic_layer_guid_t{layer_guid},
            /*per_device_op_state=*/std::nullopt,
        };
      };

      auto mk_slot = [&](TensorSlotName slot_name) -> DynamicTensorSlot {
        return DynamicTensorSlot{
            /*slot_name=*/slot_name,
            /*slot_tensor_role=*/std::nullopt,
            /*task_shard=*/std::nullopt,
        };
      };

      auto mk_value = [&](parallel_tensor_guid_t const &tensor_guid,
                          ParallelTensorShape const &shape,
                          bool create_grad = true) -> DynamicValueAttrs {
        return DynamicValueAttrs{
            /*tensor_guid=*/dynamic_tensor_guid_t{tensor_guid},
            /*parallel_tensor_shape=*/shape,
            /*create_grad=*/create_grad,
            /*subgradient_id=*/std::nullopt,
            /*shard_coord=*/std::nullopt,
            /*mapping=*/std::nullopt,
            /*accessor=*/std::nullopt,
            /*role=*/std::nullopt,
        };
      };

      DynamicNodeInvocation input_invocation = DynamicNodeInvocation{
          /*inputs=*/std::map<DynamicTensorSlot, DynamicValueAttrs>{},
          /*node=*/
          mk_node_attrs(
              input_layer.parallel_layer, input_op_attrs, input_op_mapping),
          /*outputs=*/
          std::map<DynamicTensorSlot, DynamicValueAttrs>{
              {
                  mk_slot(TensorSlotName::OUTPUT),
                  mk_value(t_input, input_pt_shape, /*create_grad=*/false),
              },
          },
      };

      DynamicNodeInvocation weight_1_invocation = DynamicNodeInvocation{
          /*inputs=*/std::map<DynamicTensorSlot, DynamicValueAttrs>{},
          /*node=*/
          mk_node_attrs(weight_1_layer.parallel_layer,
                        weight_1_op_attrs,
                        weight_1_op_mapping),
          /*outputs=*/
          std::map<DynamicTensorSlot, DynamicValueAttrs>{
              {
                  mk_slot(TensorSlotName::OUTPUT),
                  mk_value(t_weight_1, weight_1_pt_shape),
              },
          },
      };

      DynamicNodeInvocation weight_2_invocation = DynamicNodeInvocation{
          /*inputs=*/std::map<DynamicTensorSlot, DynamicValueAttrs>{},
          /*node=*/
          mk_node_attrs(weight_2_layer.parallel_layer,
                        weight_2_op_attrs,
                        weight_2_op_mapping),
          /*outputs=*/
          std::map<DynamicTensorSlot, DynamicValueAttrs>{
              {
                  mk_slot(TensorSlotName::OUTPUT),
                  mk_value(t_weight_2, weight_2_pt_shape),
              },
          },
      };

      DynamicNodeInvocation partition_input_invocation = DynamicNodeInvocation{
          /*inputs=*/std::map<DynamicTensorSlot, DynamicValueAttrs>{
              {
                  mk_slot(TensorSlotName::INPUT),
                  mk_value(t_input, input_pt_shape, /*create_grad=*/false),
              },
          },
          /*node=*/
          mk_node_attrs(partition_input_layer.parallel_layer,
                        partition_input_op_attrs,
                        partition_input_op_mapping),
          /*outputs=*/
          std::map<DynamicTensorSlot, DynamicValueAttrs>{
              {
                  mk_slot(TensorSlotName::OUTPUT),
                  mk_value(t_partitioned_input, partitioned_input_pt_shape),
              },
          },
      };

      DynamicNodeInvocation replicate_weight_1_invocation =
          DynamicNodeInvocation{
              /*inputs=*/std::map<DynamicTensorSlot, DynamicValueAttrs>{
                  {
                      mk_slot(TensorSlotName::INPUT),
                      mk_value(t_weight_1, weight_1_pt_shape),
                  },
              },
              /*node=*/
              mk_node_attrs(replicate_weight_1_layer.parallel_layer,
                            replicate_weight_1_op_attrs,
                            replicate_weight_1_op_mapping),
              /*outputs=*/
              std::map<DynamicTensorSlot, DynamicValueAttrs>{
                  {
                      mk_slot(TensorSlotName::OUTPUT),
                      mk_value(t_replicated_weight_1,
                               replicated_weight_1_pt_shape),
                  },
              },
          };

      DynamicNodeInvocation replicate_weight_2_invocation =
          DynamicNodeInvocation{
              /*inputs=*/std::map<DynamicTensorSlot, DynamicValueAttrs>{
                  {
                      mk_slot(TensorSlotName::INPUT),
                      mk_value(t_weight_2, weight_2_pt_shape),
                  },
              },
              /*node=*/
              mk_node_attrs(replicate_weight_2_layer.parallel_layer,
                            replicate_weight_2_op_attrs,
                            replicate_weight_2_op_mapping),
              /*outputs=*/
              std::map<DynamicTensorSlot, DynamicValueAttrs>{
                  {
                      mk_slot(TensorSlotName::OUTPUT),
                      mk_value(t_replicated_weight_2,
                               replicated_weight_2_pt_shape),
                  },
              },
          };

      DynamicNodeInvocation linear_1_invocation = DynamicNodeInvocation{
          /*inputs=*/std::map<DynamicTensorSlot, DynamicValueAttrs>{
              {
                  mk_slot(TensorSlotName::INPUT),
                  mk_value(t_partitioned_input, partitioned_input_pt_shape),
              },
              {
                  mk_slot(TensorSlotName::WEIGHT),
                  mk_value(t_replicated_weight_1, replicated_weight_1_pt_shape),
              },
          },
          /*node=*/
          mk_node_attrs(linear_1_layer.parallel_layer,
                        linear_1_op_attrs,
                        linear_1_op_mapping),
          /*outputs=*/
          std::map<DynamicTensorSlot, DynamicValueAttrs>{
              {
                  mk_slot(TensorSlotName::OUTPUT),
                  mk_value(t_hidden_activation, hidden_activation_pt_shape),
              },
          },
      };

      DynamicNodeInvocation linear_2_invocation = DynamicNodeInvocation{
          /*inputs=*/std::map<DynamicTensorSlot, DynamicValueAttrs>{
              {
                  mk_slot(TensorSlotName::INPUT),
                  mk_value(t_hidden_activation, hidden_activation_pt_shape),
              },
              {
                  mk_slot(TensorSlotName::WEIGHT),
                  mk_value(t_replicated_weight_2, replicated_weight_2_pt_shape),
              },
          },
          /*node=*/
          mk_node_attrs(linear_2_layer.parallel_layer,
                        linear_2_op_attrs,
                        linear_2_op_mapping),
          /*outputs=*/
          std::map<DynamicTensorSlot, DynamicValueAttrs>{
              {
                  mk_slot(TensorSlotName::OUTPUT),
                  mk_value(t_output, output_pt_shape),
              },
          },
      };

      return dynamic_open_dataflow_graph_from_invocation_set(
          /*invocations=*/{
              input_invocation,
              weight_1_invocation,
              weight_2_invocation,
              partition_input_invocation,
              replicate_weight_1_invocation,
              replicate_weight_2_invocation,
              linear_1_invocation,
              linear_2_invocation,
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
}
