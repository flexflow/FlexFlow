#include "compiler/machine_mapping/get_tensor_set_movement_across_split.h"
#include "compiler/machine_mapping/machine_view.h"
#include "compiler/machine_mapping/transitive_reduced_pcg.h"
#include "internal/cost_estimator_for_test.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "utils/containers/require_only_key.h"
#include <doctest/doctest.h>
#include <fstream>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_tensor_set_movement_across_split") {
    auto make_pcg_series_split = [](PCGBinarySPDecomposition const &lhs,
                                    PCGBinarySPDecomposition const &rhs) {
      return PCGBinarySPDecomposition{PCGBinarySeriesSplit{lhs, rhs}};
    };

    auto make_pcg_parallel_split = [](PCGBinarySPDecomposition const &lhs,
                                      PCGBinarySPDecomposition const &rhs) {
      return PCGBinarySPDecomposition{PCGBinaryParallelSplit{lhs, rhs}};
    };

    auto make_pcg_leaf_node = [](parallel_layer_guid_t const &l) {
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

    ParallelLayerAddedResult input = pcg_add_input_layer(pcg, input_shape);
    parallel_tensor_guid_t t_input =
        require_only_key(input.outputs, TensorSlotName::OUTPUT);

    ParallelLayerAttrs partition_attrs = ParallelLayerAttrs{
        /*op_attrs=*/PCGOperatorAttrs{
            RepartitionAttrs{
                /*repartition_dim=*/ff_dim_t{0_n},
                /*repartition_degree=*/2_ge2,
            },
        },
        /*name=*/std::nullopt,
    };
    ParallelLayerAddedResult partition_input = add_parallel_layer(
        pcg, partition_attrs, {{TensorSlotName::INPUT, t_input}}, {});
    parallel_tensor_guid_t t_partition_input =
        require_only_key(partition_input.outputs, TensorSlotName::OUTPUT);

    ParallelTensorShape partitioned_input_shape =
        get_parallel_tensor_shape(pcg, t_partition_input);

    ParallelLayerAttrs relu_attrs = ParallelLayerAttrs{
        /*op_attrs=*/PCGOperatorAttrs{
            ElementUnaryAttrs{
                /*op_type=*/ElementUnaryOp::RELU,
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

    ParallelLayerAddedResult relu_1 = add_parallel_layer(
        pcg, relu_attrs, {{TensorSlotName::INPUT, t_partition_input}}, {});
    parallel_tensor_guid_t t_relu_1 =
        require_only_key(relu_1.outputs, TensorSlotName::OUTPUT);
    ParallelLayerAddedResult relu_2 = add_parallel_layer(
        pcg, relu_attrs, {{TensorSlotName::INPUT, t_relu_1}}, {});

    MachineView pre_mv1 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/0_n,
            /*device_idx=*/0_n,
        },
        /*dimensions=*/
        {
            MachineViewDimension{
                stride_t{1_p},
                MachineSpecificationDimension::INTRA_NODE,
            },
        },
    };

    MachineView pre_mv2 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/1_n,
            /*device_idx=*/0_n,
        },
        /*dimensions=*/
        {
            MachineViewDimension{
                stride_t{1_p},
                MachineSpecificationDimension::INTRA_NODE,
            },
        },
    };

    MachineView post_mv1 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/2_n,
            /*device_idx=*/0_n,
        },
        /*dimensions=*/
        {
            MachineViewDimension{
                stride_t{1_p},
                MachineSpecificationDimension::INTRA_NODE,
            },
        },
    };

    MachineView post_mv2 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/3_n,
            /*device_idx=*/0_n,
        },
        /*dimensions=*/
        {
            MachineViewDimension{
                stride_t{1_p},
                MachineSpecificationDimension::INTRA_NODE,
            },
        },
    };

    auto mk_communication_edge = [](MachineView const &src_mv,
                                    nonnegative_int src_task_idx,
                                    MachineView const &dst_mv,
                                    nonnegative_int dst_task_idx) {
      ASSERT(src_task_idx < 2);
      ASSERT(dst_task_idx < 2);

      return CommunicationEdge{
          /*src=*/MachineSpaceCoordinate{
              /*node_idx=*/src_mv.start.node_idx,
              /*device_idx=*/src_task_idx,
          },
          /*dst=*/
          MachineSpaceCoordinate{
              /*node_idx=*/dst_mv.start.node_idx,
              /*device_idx=*/dst_task_idx,
          },
      };
    };

    num_bytes_t piece_size = get_piece_size_in_bytes(partitioned_input_shape);

    SUBCASE("single edge across split") {
      PCGBinarySeriesSplit split = PCGBinarySeriesSplit{
          make_pcg_leaf_node(relu_1.parallel_layer),
          make_pcg_leaf_node(relu_2.parallel_layer),
      };

      auto pre_mapping = ParallelLayerGuidObliviousMachineMapping{{
          {
              BinaryTreePath{{}},
              pre_mv1,
          },
      }};

      auto post_mapping = ParallelLayerGuidObliviousMachineMapping{{
          {
              BinaryTreePath{{}},
              post_mv1,
          },
      }};

      TensorSetMovement result = get_tensor_set_movement_across_split(
          pcg_get_transitive_reduction(pcg), split, pre_mapping, post_mapping);

      TensorSetMovement correct = TensorSetMovement{
          /*edge_to_size=*/{
              {
                  mk_communication_edge(pre_mv1, 0_n, post_mv1, 0_n),
                  piece_size,
              },
              {
                  mk_communication_edge(pre_mv1, 1_n, post_mv1, 1_n),
                  piece_size,
              },
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("does not include edges removed by transitive reduction") {
      ParallelLayerAddedResult ew_add = add_parallel_layer(
          pcg,
          ew_add_attrs,
          /*inputs=*/
          {
              {
                  TensorSlotName::LHS_INPUT,
                  require_only_key(relu_1.outputs, TensorSlotName::OUTPUT),
              },
              {
                  TensorSlotName::RHS_INPUT,
                  require_only_key(relu_2.outputs, TensorSlotName::OUTPUT),
              },
          },
          /*weights=*/{});

      PCGBinarySeriesSplit split = PCGBinarySeriesSplit{
          make_pcg_series_split(make_pcg_leaf_node(relu_1.parallel_layer),
                                make_pcg_leaf_node(relu_2.parallel_layer)),
          make_pcg_leaf_node(ew_add.parallel_layer),
      };

      auto pre_mapping = ParallelLayerGuidObliviousMachineMapping{{
          {
              BinaryTreePath{{BinaryTreePathEntry::LEFT_CHILD}},
              pre_mv2,
          },
          {
              BinaryTreePath{{BinaryTreePathEntry::RIGHT_CHILD}},
              pre_mv1,
          },
      }};

      auto post_mapping = ParallelLayerGuidObliviousMachineMapping{{
          {
              BinaryTreePath{{}},
              post_mv1,
          },
      }};

      TensorSetMovement result = get_tensor_set_movement_across_split(
          pcg_get_transitive_reduction(pcg), split, pre_mapping, post_mapping);

      TensorSetMovement correct = TensorSetMovement{
          /*edge_to_size=*/{
              {
                  mk_communication_edge(pre_mv1, 0_n, post_mv1, 0_n),
                  piece_size,
              },
              {
                  mk_communication_edge(pre_mv1, 1_n, post_mv1, 1_n),
                  piece_size,
              },
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("single tensor, multiple consumers across split") {
      ParallelLayerAddedResult relu_3 = add_parallel_layer(
          pcg,
          relu_attrs,
          /*inputs=*/
          {
              {
                  TensorSlotName::INPUT,
                  require_only_key(relu_1.outputs, TensorSlotName::OUTPUT),
              },
          },
          /*weights=*/{});

      PCGBinarySeriesSplit split = PCGBinarySeriesSplit{
          make_pcg_leaf_node(relu_1.parallel_layer),
          make_pcg_parallel_split(make_pcg_leaf_node(relu_2.parallel_layer),
                                  make_pcg_leaf_node(relu_3.parallel_layer)),
      };

      SUBCASE("consumers have same view") {
        auto pre_mapping = ParallelLayerGuidObliviousMachineMapping{{
            {
                BinaryTreePath{{}},
                pre_mv1,
            },
        }};

        auto post_mapping = ParallelLayerGuidObliviousMachineMapping{{
            {
                BinaryTreePath{{
                    BinaryTreePathEntry::LEFT_CHILD,
                }},
                post_mv1,
            },
            {
                BinaryTreePath{{
                    BinaryTreePathEntry::RIGHT_CHILD,
                }},
                post_mv1,
            },
        }};

        TensorSetMovement result = get_tensor_set_movement_across_split(
            pcg_get_transitive_reduction(pcg),
            split,
            pre_mapping,
            post_mapping);

        TensorSetMovement correct = TensorSetMovement{
            /*edge_to_size=*/{
                {
                    mk_communication_edge(pre_mv1, 0_n, post_mv1, 0_n),
                    piece_size,
                },
                {
                    mk_communication_edge(pre_mv1, 1_n, post_mv1, 1_n),
                    piece_size,
                },
            },
        };

        CHECK(result == correct);
      }

      SUBCASE("consumers have different views") {
        auto pre_mapping = ParallelLayerGuidObliviousMachineMapping{{
            {
                BinaryTreePath{{}},
                pre_mv1,
            },
        }};

        auto post_mapping = ParallelLayerGuidObliviousMachineMapping{{
            {
                BinaryTreePath{{
                    BinaryTreePathEntry::LEFT_CHILD,
                }},
                post_mv1,
            },
            {
                BinaryTreePath{{
                    BinaryTreePathEntry::RIGHT_CHILD,
                }},
                post_mv2,
            },
        }};

        TensorSetMovement result = get_tensor_set_movement_across_split(
            pcg_get_transitive_reduction(pcg),
            split,
            pre_mapping,
            post_mapping);

        TensorSetMovement correct = TensorSetMovement{
            /*edge_to_size=*/{
                {
                    mk_communication_edge(pre_mv1, 0_n, post_mv1, 0_n),
                    piece_size,
                },
                {
                    mk_communication_edge(pre_mv1, 1_n, post_mv1, 1_n),
                    piece_size,
                },
                {
                    mk_communication_edge(pre_mv1, 0_n, post_mv2, 0_n),
                    piece_size,
                },
                {
                    mk_communication_edge(pre_mv1, 1_n, post_mv2, 1_n),
                    piece_size,
                },
            },
        };

        CHECK(result == correct);
      }
    }

    SUBCASE("multiple tensors, multiple consumers across split") {
      ParallelLayerAddedResult relu_3 =
          add_parallel_layer(pcg,
                             relu_attrs,
                             /*inputs=*/
                             {
                                 {
                                     TensorSlotName::INPUT,
                                     require_only_key(partition_input.outputs,
                                                      TensorSlotName::OUTPUT),
                                 },
                             },
                             /*outputs=*/{});

      ParallelLayerAddedResult relu_4 = add_parallel_layer(
          pcg,
          ew_add_attrs,
          /*inputs=*/
          {
              {
                  TensorSlotName::LHS_INPUT,
                  require_only_key(relu_1.outputs, TensorSlotName::OUTPUT),
              },
              {TensorSlotName::RHS_INPUT,
               require_only_key(relu_3.outputs, TensorSlotName::OUTPUT)},
          },
          /*weights=*/{});

      PCGBinarySeriesSplit split = PCGBinarySeriesSplit{
          make_pcg_parallel_split(make_pcg_leaf_node(relu_1.parallel_layer),
                                  make_pcg_leaf_node(relu_3.parallel_layer)),
          make_pcg_parallel_split(make_pcg_leaf_node(relu_2.parallel_layer),
                                  make_pcg_leaf_node(relu_4.parallel_layer)),
      };

      auto mk_pre_mapping = [](MachineView const &src1_mv,
                               MachineView const &src2_mv) {
        return ParallelLayerGuidObliviousMachineMapping{{
            {
                BinaryTreePath{{
                    BinaryTreePathEntry::LEFT_CHILD,
                }},
                src1_mv,
            },
            {
                BinaryTreePath{{
                    BinaryTreePathEntry::RIGHT_CHILD,
                }},
                src2_mv,
            },
        }};
      };

      auto mk_post_mapping = [](MachineView const &dst1_mv,
                                MachineView const &dst2_mv) {
        return ParallelLayerGuidObliviousMachineMapping{{
            {
                BinaryTreePath{{
                    BinaryTreePathEntry::LEFT_CHILD,
                }},
                dst1_mv,
            },
            {
                BinaryTreePath{{
                    BinaryTreePathEntry::RIGHT_CHILD,
                }},
                dst2_mv,
            },
        }};
      };

      SUBCASE(
          "producers have different views and consumers have different views") {
        ParallelLayerGuidObliviousMachineMapping pre_mapping =
            mk_pre_mapping(pre_mv1, pre_mv2);
        ParallelLayerGuidObliviousMachineMapping post_mapping =
            mk_post_mapping(post_mv1, post_mv2);

        TensorSetMovement result = get_tensor_set_movement_across_split(
            pcg_get_transitive_reduction(pcg),
            split,
            pre_mapping,
            post_mapping);

        TensorSetMovement correct = TensorSetMovement{
            /*edge_to_size=*/{
                {
                    mk_communication_edge(pre_mv1, 0_n, post_mv1, 0_n),
                    piece_size,
                },
                {
                    mk_communication_edge(pre_mv1, 1_n, post_mv1, 1_n),
                    piece_size,
                },
                {
                    mk_communication_edge(pre_mv1, 0_n, post_mv2, 0_n),
                    piece_size,
                },
                {
                    mk_communication_edge(pre_mv1, 1_n, post_mv2, 1_n),
                    piece_size,
                },
                {
                    mk_communication_edge(pre_mv2, 0_n, post_mv2, 0_n),
                    piece_size,
                },
                {
                    mk_communication_edge(pre_mv2, 1_n, post_mv2, 1_n),
                    piece_size,
                },
            },
        };

        CHECK(result == correct);
      }

      SUBCASE(
          "producers have different views and consumers have the same view") {
        ParallelLayerGuidObliviousMachineMapping pre_mapping =
            mk_pre_mapping(pre_mv1, pre_mv2);
        ParallelLayerGuidObliviousMachineMapping post_mapping =
            mk_post_mapping(post_mv1, post_mv1);

        TensorSetMovement result = get_tensor_set_movement_across_split(
            pcg_get_transitive_reduction(pcg),
            split,
            pre_mapping,
            post_mapping);

        TensorSetMovement correct = TensorSetMovement{
            /*edge_to_size=*/{
                {
                    mk_communication_edge(pre_mv1, 0_n, post_mv1, 0_n),
                    piece_size,
                },
                {
                    mk_communication_edge(pre_mv1, 1_n, post_mv1, 1_n),
                    piece_size,
                },
                {
                    mk_communication_edge(pre_mv2, 0_n, post_mv1, 0_n),
                    piece_size,
                },
                {
                    mk_communication_edge(pre_mv2, 1_n, post_mv1, 1_n),
                    piece_size,
                },
            },
        };

        CHECK(result == correct);
      }

      SUBCASE(
          "producers have the same view and consumers have different views") {
        ParallelLayerGuidObliviousMachineMapping pre_mapping =
            mk_pre_mapping(pre_mv1, pre_mv1);
        ParallelLayerGuidObliviousMachineMapping post_mapping =
            mk_post_mapping(post_mv1, post_mv2);

        TensorSetMovement result = get_tensor_set_movement_across_split(
            pcg_get_transitive_reduction(pcg),
            split,
            pre_mapping,
            post_mapping);

        TensorSetMovement correct = TensorSetMovement{
            /*edge_to_size=*/{
                {
                    mk_communication_edge(pre_mv1, 0_n, post_mv1, 0_n),
                    piece_size,
                },
                {
                    mk_communication_edge(pre_mv1, 1_n, post_mv1, 1_n),
                    piece_size,
                },
                {
                    mk_communication_edge(pre_mv1, 0_n, post_mv2, 0_n),
                    piece_size + piece_size,
                },
                {
                    mk_communication_edge(pre_mv1, 1_n, post_mv2, 1_n),
                    piece_size + piece_size,
                },
            },
        };

        CHECK(result == correct);
      }

      SUBCASE("producers have the same view and consumers have the same view") {
        ParallelLayerGuidObliviousMachineMapping pre_mapping =
            mk_pre_mapping(pre_mv1, pre_mv1);
        ParallelLayerGuidObliviousMachineMapping post_mapping =
            mk_post_mapping(post_mv1, post_mv1);

        TensorSetMovement result = get_tensor_set_movement_across_split(
            pcg_get_transitive_reduction(pcg),
            split,
            pre_mapping,
            post_mapping);

        TensorSetMovement correct = TensorSetMovement{
            /*edge_to_size=*/{
                {
                    mk_communication_edge(pre_mv1, 0_n, post_mv1, 0_n),
                    piece_size + piece_size,
                },
                {
                    mk_communication_edge(pre_mv1, 1_n, post_mv1, 1_n),
                    piece_size + piece_size,
                },
            },
        };

        CHECK(result == correct);
      }

      SUBCASE("all producers and consumers have the same view") {
        ParallelLayerGuidObliviousMachineMapping pre_mapping =
            mk_pre_mapping(pre_mv1, pre_mv1);
        ParallelLayerGuidObliviousMachineMapping post_mapping =
            mk_post_mapping(pre_mv1, pre_mv1);

        TensorSetMovement result = get_tensor_set_movement_across_split(
            pcg_get_transitive_reduction(pcg),
            split,
            pre_mapping,
            post_mapping);

        TensorSetMovement correct = TensorSetMovement{
            /*edge_to_size=*/{{}},
        };

        CHECK(result == correct);
      }

      SUBCASE("producers and one consumer share the same view") {
        ParallelLayerGuidObliviousMachineMapping pre_mapping =
            mk_pre_mapping(pre_mv1, pre_mv1);
        ParallelLayerGuidObliviousMachineMapping post_mapping =
            mk_post_mapping(post_mv1, pre_mv1);

        TensorSetMovement result = get_tensor_set_movement_across_split(
            pcg_get_transitive_reduction(pcg),
            split,
            pre_mapping,
            post_mapping);

        TensorSetMovement correct = TensorSetMovement{
            /*edge_to_size=*/{
                {
                    mk_communication_edge(pre_mv1, 0_n, post_mv1, 0_n),
                    piece_size,
                },
                {
                    mk_communication_edge(pre_mv1, 1_n, post_mv1, 1_n),
                    piece_size,
                },
            },
        };

        CHECK(result == correct);
      }
    }
  }
}
