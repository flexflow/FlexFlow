#include "compiler/machine_mapping/abstracted_tensor_set_movement/get_abstracted_tensor_set_movement_across_split.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_single_tensor_communication.dtg.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_single_tensor_movement.h"
#include "compiler/machine_mapping/transitive_reduced_pcg.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "utils/containers/get_only.h"
#include "utils/containers/require_only_key.h"
#include "utils/full_binary_tree/binary_tree_path.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_abstracted_single_tensor_movement_along_edge") {
    ParallelComputationGraph pcg = empty_parallel_computation_graph();

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered{
                10_p,
                12_p,
            },
        },
        DataType::FLOAT,
    };

    ParallelTensorShape par_input_shape = lift_to_parallel(input_shape);

    ParallelLayerAttrs partition_attrs = ParallelLayerAttrs{
        /*op_attrs=*/PCGOperatorAttrs{
            RepartitionAttrs{
                /*repartition_dim=*/ff_dim_t{0_n},
                /*repartition_degree=*/2_p,
            },
        },
        /*name=*/std::nullopt,
    };

    ParallelLayerAttrs relu_attrs = ParallelLayerAttrs{
        /*op_attrs=*/PCGOperatorAttrs{
            ElementUnaryAttrs{
                /*op_type=*/OperatorType::RELU,
                /*scalar=*/std::nullopt,
            },
        },
        /*name=*/std::nullopt,
    };

    ParallelLayerAddedResult input = pcg_add_input_layer(pcg, input_shape);
    parallel_tensor_guid_t t_input =
        require_only_key(input.outputs, TensorSlotName::OUTPUT);
    ParallelLayerAddedResult partition_input = add_parallel_layer(
        pcg, partition_attrs, {{TensorSlotName::INPUT, t_input}}, {});
    parallel_tensor_guid_t t_partition_input =
        require_only_key(partition_input.outputs, TensorSlotName::OUTPUT);

    ParallelLayerAddedResult layer_1 = add_parallel_layer(
        pcg, relu_attrs, {{TensorSlotName::INPUT, t_partition_input}}, {});
    parallel_tensor_guid_t t_layer_1 =
        require_only_key(layer_1.outputs, TensorSlotName::OUTPUT);
    ParallelLayerAddedResult layer_2 = add_parallel_layer(
        pcg, relu_attrs, {{TensorSlotName::INPUT, t_layer_1}}, {});

    ParallelComputationGraphEdge edge =
        get_only(get_pcg_edges_from_layer_to_layer(
            /*pcg=*/pcg,
            /*src=*/layer_1.parallel_layer,
            /*dst=*/layer_2.parallel_layer));

    BinaryTreePath src_path = BinaryTreePath{{}};
    BinaryTreePath dst_path = BinaryTreePath{{}};

    AbstractedSingleTensorMovement result =
        get_abstracted_single_tensor_movement_along_edge(
            pcg, edge, src_path, dst_path);

    num_bytes_t shard_size =
        get_piece_size_in_bytes(get_parallel_tensor_shape(pcg, t_layer_1));

    auto mk_single_tensor_communication =
        [&](nonnegative_int src_coord,
            nonnegative_int dst_coord) -> AbstractedSingleTensorCommunication {
      return AbstractedSingleTensorCommunication{
          /*edge=*/AbstractedSingleTensorCommunicationEdge{
              /*src_coord=*/TaskSpaceCoordinate{OrthotopeCoord{{src_coord}}},
              /*dst=*/
              AbstractedDevice{
                  /*operator_tree_path=*/dst_path,
                  /*task_space_coordinate=*/
                  TaskSpaceCoordinate{OrthotopeCoord{{dst_coord}}},
              },
          },
          /*size=*/shard_size,
      };
    };

    AbstractedSingleTensorMovement correct =
        abstracted_single_tensor_movement_from_communications(
            /*src_op_tree_path=*/src_path,
            /*communications=*/{
                mk_single_tensor_communication(0_n, 0_n),
                mk_single_tensor_communication(1_n, 1_n),
            });

    CHECK(result == correct);
  }

  TEST_CASE("get_abstracted_tensor_set_movement_across_split") {
    auto make_series_split = [](PCGBinarySPDecomposition const &lhs,
                                PCGBinarySPDecomposition const &rhs) {
      return PCGBinarySPDecomposition{PCGBinarySeriesSplit{lhs, rhs}};
    };

    auto make_parallel_split = [](PCGBinarySPDecomposition const &lhs,
                                  PCGBinarySPDecomposition const &rhs) {
      return PCGBinarySPDecomposition{PCGBinaryParallelSplit{lhs, rhs}};
    };

    auto make_leaf = [](parallel_layer_guid_t const &l) {
      return PCGBinarySPDecomposition{l};
    };

    ParallelComputationGraph pcg = empty_parallel_computation_graph();

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered{
                10_p,
                12_p,
            },
        },
        DataType::FLOAT,
    };

    ParallelTensorShape par_input_shape = lift_to_parallel(input_shape);

    ParallelLayerAttrs partition_attrs = ParallelLayerAttrs{
        /*op_attrs=*/PCGOperatorAttrs{
            RepartitionAttrs{
                /*repartition_dim=*/ff_dim_t{0_n},
                /*repartition_degree=*/2_p,
            },
        },
        /*name=*/std::nullopt,
    };

    ParallelLayerAttrs relu_attrs = ParallelLayerAttrs{
        /*op_attrs=*/PCGOperatorAttrs{
            ElementUnaryAttrs{
                /*op_type=*/OperatorType::RELU,
                /*scalar=*/std::nullopt,
            },
        },
        /*name=*/std::nullopt,
    };

    ParallelLayerAttrs ew_add_attrs = ParallelLayerAttrs{
        /*op_attrs=*/PCGOperatorAttrs{
            ElementBinaryAttrs{
                /*op=*/ElementBinaryOp::ADD,
                /*compute_type=*/DataType::FLOAT,
                /*should_broadcast_lhs=*/false,
                /*should_broadcast_rhs=*/false,
            },
        },
        /*name=*/std::nullopt,
    };

    auto mk_task_space_coord = [&](nonnegative_int coord) {
      return TaskSpaceCoordinate{
          OrthotopeCoord{{
              coord,
          }},
      };
    };

    auto mk_abstracted_device = [&](BinaryTreePath const &path,
                                    nonnegative_int coord) {
      return AbstractedDevice{
          /*operator_tree_path=*/path,
          /*task_space_coordinate=*/mk_task_space_coord(coord),
      };
    };

    SUBCASE("no edges across split") {
      ParallelLayerAddedResult input1 = pcg_add_input_layer(pcg, input_shape);
      parallel_tensor_guid_t t_input1 =
          require_only_key(input1.outputs, TensorSlotName::OUTPUT);
      ParallelLayerAddedResult partition_input1 = add_parallel_layer(
          pcg, partition_attrs, {{TensorSlotName::INPUT, t_input1}}, {});

      ParallelLayerAddedResult input2 = pcg_add_input_layer(pcg, input_shape);
      parallel_tensor_guid_t t_input2 =
          require_only_key(input2.outputs, TensorSlotName::OUTPUT);
      ParallelLayerAddedResult partition_input2 = add_parallel_layer(
          pcg, partition_attrs, {{TensorSlotName::INPUT, t_input2}}, {});

      PCGBinarySeriesSplit split = PCGBinarySeriesSplit{
          make_series_split(make_leaf(input1.parallel_layer),
                            make_leaf(partition_input1.parallel_layer)),
          make_series_split(make_leaf(input2.parallel_layer),
                            make_leaf(partition_input2.parallel_layer)),
      };

      AbstractedTensorSetMovement result =
          get_abstracted_tensor_set_movement_across_split(
              pcg_get_transitive_reduction(pcg), split);

      AbstractedTensorSetMovement correct = AbstractedTensorSetMovement{
          /*single_tensor_movements=*/{},
      };

      CHECK(result == correct);
    }

    SUBCASE("single edge across split") {
      ParallelLayerAddedResult input = pcg_add_input_layer(pcg, input_shape);
      parallel_tensor_guid_t t_input =
          require_only_key(input.outputs, TensorSlotName::OUTPUT);
      ParallelLayerAddedResult partition_input = add_parallel_layer(
          pcg, partition_attrs, {{TensorSlotName::INPUT, t_input}}, {});
      parallel_tensor_guid_t t_partition_input =
          require_only_key(partition_input.outputs, TensorSlotName::OUTPUT);

      ParallelLayerAddedResult layer_1 = add_parallel_layer(
          pcg, relu_attrs, {{TensorSlotName::INPUT, t_partition_input}}, {});
      parallel_tensor_guid_t t_layer_1 =
          require_only_key(layer_1.outputs, TensorSlotName::OUTPUT);
      ParallelLayerAddedResult layer_2 = add_parallel_layer(
          pcg, relu_attrs, {{TensorSlotName::INPUT, t_layer_1}}, {});

      PCGBinarySeriesSplit split = PCGBinarySeriesSplit{
          make_series_split(
              make_series_split(make_leaf(input.parallel_layer),
                                make_leaf(partition_input.parallel_layer)),
              make_leaf(layer_1.parallel_layer)),
          make_leaf(layer_2.parallel_layer),
      };

      AbstractedTensorSetMovement result =
          get_abstracted_tensor_set_movement_across_split(
              pcg_get_transitive_reduction(pcg), split);

      BinaryTreePath src_path = BinaryTreePath{{
          BinaryTreePathEntry::RIGHT_CHILD,
      }};

      BinaryTreePath dst_path = BinaryTreePath{{}};

      auto mk_abstracted_edge = [&](nonnegative_int src_coord,
                                    nonnegative_int dst_coord) {
        return AbstractedSingleTensorCommunicationEdge{
            /*src=*/mk_task_space_coord(src_coord),
            /*dst=*/mk_abstracted_device(dst_path, dst_coord),
        };
      };

      num_bytes_t shard_size = get_size_in_bytes(
          get_reduced_shape(get_parallel_tensor_shape(pcg, t_layer_1)));

      AbstractedTensorSetMovement correct = AbstractedTensorSetMovement{
          /*single_tensor_movements=*/{
              AbstractedSingleTensorMovement{
                  /*src_op_tree_path=*/src_path,
                  /*edge_to_size=*/
                  {
                      {mk_abstracted_edge(0_n, 0_n), shard_size},
                      {mk_abstracted_edge(1_n, 1_n), shard_size},
                  },
              },
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("does not include edges removed by transitive reduction") {
      ParallelLayerAddedResult input = pcg_add_input_layer(pcg, input_shape);
      parallel_tensor_guid_t t_input =
          require_only_key(input.outputs, TensorSlotName::OUTPUT);
      ParallelLayerAddedResult partition_input = add_parallel_layer(
          pcg, partition_attrs, {{TensorSlotName::INPUT, t_input}}, {});
      parallel_tensor_guid_t t_partition_input =
          require_only_key(partition_input.outputs, TensorSlotName::OUTPUT);

      ParallelLayerAddedResult layer_1 = add_parallel_layer(
          pcg, relu_attrs, {{TensorSlotName::INPUT, t_partition_input}}, {});
      parallel_tensor_guid_t t_layer_1 =
          require_only_key(layer_1.outputs, TensorSlotName::OUTPUT);

      ParallelLayerAddedResult layer_2 = add_parallel_layer(
          pcg, relu_attrs, {{TensorSlotName::INPUT, t_layer_1}}, {});
      parallel_tensor_guid_t t_layer_2 =
          require_only_key(layer_2.outputs, TensorSlotName::OUTPUT);

      ParallelLayerAddedResult layer_3 =
          add_parallel_layer(pcg,
                             ew_add_attrs,
                             /*inputs=*/
                             {
                                 {
                                     TensorSlotName::LHS_INPUT,
                                     t_layer_1,
                                 },
                                 {
                                     TensorSlotName::RHS_INPUT,
                                     t_layer_2,
                                 },
                             },
                             /*weights=*/{});

      PCGBinarySeriesSplit split = PCGBinarySeriesSplit{
          make_series_split(
              make_series_split(make_leaf(input.parallel_layer),
                                make_leaf(partition_input.parallel_layer)),
              make_series_split(make_leaf(layer_1.parallel_layer),
                                make_leaf(layer_2.parallel_layer))),
          make_leaf(layer_3.parallel_layer),
      };

      AbstractedTensorSetMovement result =
          get_abstracted_tensor_set_movement_across_split(
              pcg_get_transitive_reduction(pcg), split);

      BinaryTreePath src_path = BinaryTreePath{{
          BinaryTreePathEntry::RIGHT_CHILD,
          BinaryTreePathEntry::RIGHT_CHILD,
      }};

      BinaryTreePath dst_path = BinaryTreePath{{}};

      auto mk_abstracted_edge = [&](nonnegative_int src_coord,
                                    nonnegative_int dst_coord) {
        return AbstractedSingleTensorCommunicationEdge{
            /*src=*/mk_task_space_coord(src_coord),
            /*dst=*/mk_abstracted_device(dst_path, dst_coord),
        };
      };

      num_bytes_t shard_size = get_size_in_bytes(
          get_reduced_shape(get_parallel_tensor_shape(pcg, t_layer_2)));

      AbstractedTensorSetMovement correct = AbstractedTensorSetMovement{
          /*single_tensor_movements=*/{
              AbstractedSingleTensorMovement{
                  /*src_op_tree_path=*/src_path,
                  /*edge_to_size=*/
                  {
                      {mk_abstracted_edge(0_n, 0_n), shard_size},
                      {mk_abstracted_edge(1_n, 1_n), shard_size},
                  },
              },
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("single tensor, multiple consumers across split") {
      ParallelLayerAddedResult input = pcg_add_input_layer(pcg, input_shape);
      parallel_tensor_guid_t t_input =
          require_only_key(input.outputs, TensorSlotName::OUTPUT);
      ParallelLayerAddedResult partition_input = add_parallel_layer(
          pcg, partition_attrs, {{TensorSlotName::INPUT, t_input}}, {});
      parallel_tensor_guid_t t_partition_input =
          require_only_key(partition_input.outputs, TensorSlotName::OUTPUT);

      ParallelLayerAddedResult layer_1 = add_parallel_layer(
          pcg, relu_attrs, {{TensorSlotName::INPUT, t_partition_input}}, {});
      parallel_tensor_guid_t t_layer_1 =
          require_only_key(layer_1.outputs, TensorSlotName::OUTPUT);

      ParallelLayerAddedResult layer_2 = add_parallel_layer(
          pcg, relu_attrs, {{TensorSlotName::INPUT, t_layer_1}}, {});

      ParallelLayerAddedResult layer_3 = add_parallel_layer(
          pcg, relu_attrs, {{TensorSlotName::INPUT, t_layer_1}}, {});

      PCGBinarySeriesSplit split = PCGBinarySeriesSplit{
          make_series_split(
              make_series_split(make_leaf(input.parallel_layer),
                                make_leaf(partition_input.parallel_layer)),
              make_leaf(layer_1.parallel_layer)),
          make_parallel_split(make_leaf(layer_2.parallel_layer),
                              make_leaf(layer_3.parallel_layer)),
      };

      AbstractedTensorSetMovement result =
          get_abstracted_tensor_set_movement_across_split(
              pcg_get_transitive_reduction(pcg), split);

      BinaryTreePath src_path = BinaryTreePath{{
          BinaryTreePathEntry::RIGHT_CHILD,
      }};

      BinaryTreePath dst1_path = BinaryTreePath{{
          BinaryTreePathEntry::LEFT_CHILD,
      }};

      BinaryTreePath dst2_path = BinaryTreePath{{
          BinaryTreePathEntry::RIGHT_CHILD,
      }};

      auto mk_abstracted_edge = [&](nonnegative_int src_coord,
                                    BinaryTreePath dst_path,
                                    nonnegative_int dst_coord) {
        return AbstractedSingleTensorCommunicationEdge{
            /*src=*/mk_task_space_coord(src_coord),
            /*dst=*/mk_abstracted_device(dst_path, dst_coord),
        };
      };

      num_bytes_t shard_size = get_size_in_bytes(
          get_reduced_shape(get_parallel_tensor_shape(pcg, t_layer_1)));

      AbstractedTensorSetMovement correct = AbstractedTensorSetMovement{
          /*single_tensor_movements=*/{
              AbstractedSingleTensorMovement{
                  /*src_op_tree_path=*/src_path,
                  /*edge_to_size=*/
                  {
                      {mk_abstracted_edge(0_n, dst1_path, 0_n), shard_size},
                      {mk_abstracted_edge(1_n, dst1_path, 1_n), shard_size},
                      {mk_abstracted_edge(0_n, dst2_path, 0_n), shard_size},
                      {mk_abstracted_edge(1_n, dst2_path, 1_n), shard_size},
                  },
              },
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("multiple tensors, multiple consumers across split") {
      ParallelLayerAddedResult input = pcg_add_input_layer(pcg, input_shape);
      parallel_tensor_guid_t t_input =
          require_only_key(input.outputs, TensorSlotName::OUTPUT);
      ParallelLayerAddedResult partition_input = add_parallel_layer(
          pcg, partition_attrs, {{TensorSlotName::INPUT, t_input}}, {});
      parallel_tensor_guid_t t_partition_input =
          require_only_key(partition_input.outputs, TensorSlotName::OUTPUT);

      ParallelLayerAddedResult layer_1 = add_parallel_layer(
          pcg, relu_attrs, {{TensorSlotName::INPUT, t_partition_input}}, {});
      parallel_tensor_guid_t t_layer_1 =
          require_only_key(layer_1.outputs, TensorSlotName::OUTPUT);

      ParallelLayerAddedResult layer_2 = add_parallel_layer(
          pcg, relu_attrs, {{TensorSlotName::INPUT, t_partition_input}}, {});
      parallel_tensor_guid_t t_layer_2 =
          require_only_key(layer_2.outputs, TensorSlotName::OUTPUT);

      ParallelLayerAddedResult layer_3 = add_parallel_layer(
          pcg, relu_attrs, {{TensorSlotName::INPUT, t_layer_1}}, {});

      ParallelLayerAddedResult layer_4 =
          add_parallel_layer(pcg,
                             ew_add_attrs,
                             /*inputs=*/
                             {
                                 {
                                     TensorSlotName::LHS_INPUT,
                                     t_layer_1,
                                 },
                                 {
                                     TensorSlotName::RHS_INPUT,
                                     t_layer_2,
                                 },
                             },
                             /*weights=*/{});

      PCGBinarySeriesSplit split = PCGBinarySeriesSplit{
          make_series_split(
              make_series_split(make_leaf(input.parallel_layer),
                                make_leaf(partition_input.parallel_layer)),
              make_parallel_split(make_leaf(layer_1.parallel_layer),
                                  make_leaf(layer_2.parallel_layer))),
          make_parallel_split(make_leaf(layer_3.parallel_layer),
                              make_leaf(layer_4.parallel_layer))};

      AbstractedTensorSetMovement result =
          get_abstracted_tensor_set_movement_across_split(
              pcg_get_transitive_reduction(pcg), split);

      BinaryTreePath src1_path = BinaryTreePath{{
          BinaryTreePathEntry::RIGHT_CHILD,
          BinaryTreePathEntry::LEFT_CHILD,
      }};

      BinaryTreePath src2_path = BinaryTreePath{{
          BinaryTreePathEntry::RIGHT_CHILD,
          BinaryTreePathEntry::RIGHT_CHILD,
      }};

      BinaryTreePath dst1_path = BinaryTreePath{{
          BinaryTreePathEntry::LEFT_CHILD,
      }};

      BinaryTreePath dst2_path = BinaryTreePath{{
          BinaryTreePathEntry::RIGHT_CHILD,
      }};

      auto mk_abstracted_edge = [&](nonnegative_int src_coord,
                                    BinaryTreePath dst_path,
                                    nonnegative_int dst_coord) {
        return AbstractedSingleTensorCommunicationEdge{
            /*src=*/mk_task_space_coord(src_coord),
            /*dst=*/mk_abstracted_device(dst_path, dst_coord),
        };
      };

      num_bytes_t t1_shard_size = get_size_in_bytes(
          get_reduced_shape(get_parallel_tensor_shape(pcg, t_layer_1)));
      num_bytes_t t2_shard_size = get_size_in_bytes(
          get_reduced_shape(get_parallel_tensor_shape(pcg, t_layer_2)));

      AbstractedTensorSetMovement correct = AbstractedTensorSetMovement{
          /*single_tensor_movements=*/{
              AbstractedSingleTensorMovement{
                  /*src_op_tree_path=*/src1_path,
                  /*edge_to_size=*/
                  {
                      {mk_abstracted_edge(0_n, dst1_path, 0_n), t1_shard_size},
                      {mk_abstracted_edge(1_n, dst1_path, 1_n), t1_shard_size},
                      {mk_abstracted_edge(0_n, dst2_path, 0_n), t1_shard_size},
                      {mk_abstracted_edge(1_n, dst2_path, 1_n), t1_shard_size},
                  },
              },
              AbstractedSingleTensorMovement{
                  /*src_op_tree_path=*/src2_path,
                  /*edge_to_size=*/
                  {
                      {mk_abstracted_edge(0_n, dst2_path, 0_n), t2_shard_size},
                      {mk_abstracted_edge(1_n, dst2_path, 1_n), t2_shard_size},
                  },
              },
          },
      };

      CHECK(result == correct);
    }
  }
}
