#include "compiler/machine_mapping/machine_mapping_problem_tree/get_machine_mapping_problem_tree.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/machine_mapping_problem_tree.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "utils/containers/get_only.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_machine_mapping_problem_tree") {
    auto pcg_make_leaf = [](parallel_layer_guid_t const &l) {
      return PCGBinarySPDecomposition{l};
    };

    auto pcg_make_series = [](PCGBinarySPDecomposition const &lhs,
                              PCGBinarySPDecomposition const &rhs) {
      return PCGBinarySPDecomposition{
          PCGBinarySeriesSplit{
              lhs,
              rhs,
          },
      };
    };

    auto pcg_make_parallel = [](PCGBinarySPDecomposition const &lhs,
                                PCGBinarySPDecomposition const &rhs) {
      return PCGBinarySPDecomposition{
          PCGBinaryParallelSplit{
              lhs,
              rhs,
          },
      };
    };

    auto mm_problem_tree_make_leaf = [](UnmappedOpCostEstimateKey const &k) {
      return MachineMappingProblemTree{k};
    };

    auto mm_problem_tree_make_series =
        [](AbstractedTensorSetMovement const &tensor_set_movement,
           MachineMappingProblemTree const &lhs,
           MachineMappingProblemTree const &rhs) {
          return MachineMappingProblemTree{
              MMProblemTreeSeriesSplit{
                  tensor_set_movement,
                  lhs,
                  rhs,
              },
          };
        };

    auto mm_problem_tree_make_parallel =
        [](MachineMappingProblemTree const &lhs,
           MachineMappingProblemTree const &rhs) {
          return MachineMappingProblemTree{
              MMProblemTreeParallelSplit{
                  lhs,
                  rhs,
              },
          };
        };

    ParallelComputationGraph pcg = empty_parallel_computation_graph();

    ParallelTensorShape input_shape = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{10, 1},
            },
            ReplicaParallelDimSet{
                SumDegree{1},
                DiscardCopyDegree{1},
            },
        },
        DataType::FLOAT,
    };

    auto make_output_attrs = [](ParallelTensorShape const &shape) {
      return ParallelTensorAttrs{
          /*shape=*/shape,
          /*sync_type=*/std::nullopt,
          /*initializer=*/std::nullopt,
          /*create_gradients=*/CreateGrad::YES,
      };
    };

    auto make_layer_attrs = [](PCGOperatorAttrs const &op_attrs) {
      return ParallelLayerAttrs{
          /*op_attrs=*/op_attrs,
          /*name=*/std::nullopt,
      };
    };

    PCGOperatorAttrs input_attrs = PCGOperatorAttrs{InputAttrs{}};

    auto make_input_key =
        [&](ParallelTensorShape const &parallel_tensor_shape) {
          return UnmappedOpCostEstimateKey{
              /*op_attrs=*/input_attrs,
              /*input_shapes=*/{},
              /*weight_shapes=*/{},
              /*output_shapes=*/{parallel_tensor_shape},
          };
        };

    SUBCASE("single layer") {
      ParallelLayerAddedResult input_added = add_parallel_layer(
          pcg,
          /*layer_attrs=*/make_layer_attrs(input_attrs),
          /*inputs=*/{},
          /*output_labels=*/{make_output_attrs(input_shape)});
      parallel_layer_guid_t input_layer = input_added.parallel_layer;

      UnmappedOpCostEstimateKey input_key = make_input_key(input_shape);

      PCGBinarySPDecomposition sp_decomposition =
          PCGBinarySPDecomposition{input_layer};

      MachineMappingProblemTree result =
          get_machine_mapping_problem_tree(pcg, sp_decomposition);
      MachineMappingProblemTree correct = MachineMappingProblemTree{input_key};

      CHECK(result == correct);
    }

    SUBCASE("two layers in series") {
      ParallelLayerAddedResult input_added = add_parallel_layer(
          pcg,
          /*layer_attrs=*/make_layer_attrs(input_attrs),
          /*inputs=*/{},
          /*output_labels=*/{make_output_attrs(input_shape)});
      parallel_layer_guid_t input_layer = input_added.parallel_layer;
      parallel_tensor_guid_t input = get_only(input_added.outputs);

      UnmappedOpCostEstimateKey input_key = make_input_key(input_shape);

      PCGOperatorAttrs relu_attrs = PCGOperatorAttrs{
          ElementUnaryAttrs{
              /*op_type=*/OperatorType::RELU,
              /*scalar=*/std::nullopt,
          },
      };
      ParallelTensorShape relu_output_shape = input_shape;
      ParallelLayerAddedResult relu_added =
          add_parallel_layer(pcg,
                             make_layer_attrs(relu_attrs),
                             {input},
                             {make_output_attrs(relu_output_shape)});
      parallel_layer_guid_t relu_layer = relu_added.parallel_layer;
      parallel_tensor_guid_t relu_output = get_only(relu_added.outputs);

      UnmappedOpCostEstimateKey relu_key = UnmappedOpCostEstimateKey{
          /*op_attrs=*/relu_attrs,
          /*input_shapes=*/{input_shape},
          /*weight_shapes=*/{},
          /*output_shapes=*/{relu_output_shape},
      };

      PCGBinarySPDecomposition sp_decomposition = pcg_make_series(
          pcg_make_leaf(input_layer), pcg_make_leaf(relu_layer));

      MachineMappingProblemTree result =
          get_machine_mapping_problem_tree(pcg, sp_decomposition);

      MachineMappingProblemTree correct = mm_problem_tree_make_series(
          AbstractedTensorSetMovement{{
              AbstractedSingleTensorMovement{
                  /*parallel_tensor_shape=*/input_shape,
                  /*src_machine_views=*/
                  {
                      BinaryTreePath{{}},
                  },
                  /*dst_machine_views=*/
                  {
                      BinaryTreePath{{}},
                  },
              },
          }},
          mm_problem_tree_make_leaf(input_key),
          mm_problem_tree_make_leaf(relu_key));

      CHECK(result == correct);
    }

    SUBCASE("two layers in parallel") {
      ParallelLayerAddedResult input1_added =
          pcg_add_input_layer(pcg, input_shape);
      parallel_layer_guid_t input1_layer = input1_added.parallel_layer;
      UnmappedOpCostEstimateKey input1_key = make_input_key(input_shape);

      ParallelLayerAddedResult input2_added =
          pcg_add_input_layer(pcg, input_shape);
      parallel_layer_guid_t input2_layer = input2_added.parallel_layer;
      UnmappedOpCostEstimateKey input2_key = make_input_key(input_shape);

      PCGBinarySPDecomposition sp_decomposition = pcg_make_parallel(
          pcg_make_leaf(input1_layer), pcg_make_leaf(input2_layer));

      MachineMappingProblemTree result =
          get_machine_mapping_problem_tree(pcg, sp_decomposition);

      MachineMappingProblemTree correct =
          mm_problem_tree_make_parallel(mm_problem_tree_make_leaf(input1_key),
                                        mm_problem_tree_make_leaf(input2_key));

      CHECK(result == correct);
    }

    SUBCASE("multiple tensors across split") {
      ParallelLayerAddedResult input1_added =
          pcg_add_input_layer(pcg, input_shape);
      parallel_layer_guid_t input1_layer = input1_added.parallel_layer;
      parallel_tensor_guid_t input1_tensor = get_only(input1_added.outputs);
      UnmappedOpCostEstimateKey input1_key = make_input_key(input_shape);

      ParallelLayerAddedResult input2_added =
          pcg_add_input_layer(pcg, input_shape);
      parallel_layer_guid_t input2_layer = input2_added.parallel_layer;
      parallel_tensor_guid_t input2_tensor = get_only(input2_added.outputs);
      UnmappedOpCostEstimateKey input2_key = make_input_key(input_shape);

      PCGOperatorAttrs ew_op_attrs = PCGOperatorAttrs{
          ElementBinaryAttrs{
              /*type=*/OperatorType::EW_ADD,
              /*compute_type=*/DataType::FLOAT,
              /*should_broadcast_lhs=*/false,
              /*should_broadcast_rhs=*/false,
          },
      };
      ParallelTensorShape ew_op_output_shape = input_shape;
      ParallelLayerAddedResult ew_op_added =
          add_parallel_layer(pcg,
                             make_layer_attrs(ew_op_attrs),
                             {input1_tensor, input2_tensor},
                             {make_output_attrs(ew_op_output_shape)});
      parallel_layer_guid_t ew_op_layer = ew_op_added.parallel_layer;
      UnmappedOpCostEstimateKey ew_op_key = UnmappedOpCostEstimateKey{
          /*op_attrs=*/ew_op_attrs,
          /*input_shapes=*/{input_shape, input_shape},
          /*weight_shapes=*/{},
          /*output_shapes=*/{ew_op_output_shape},
      };

      PCGBinarySPDecomposition sp_decomposition =
          pcg_make_series(pcg_make_parallel(pcg_make_leaf(input1_layer),
                                            pcg_make_leaf(input2_layer)),
                          pcg_make_leaf(ew_op_layer));

      MachineMappingProblemTree result =
          get_machine_mapping_problem_tree(pcg, sp_decomposition);

      MachineMappingProblemTree correct = mm_problem_tree_make_series(
          AbstractedTensorSetMovement{{
              AbstractedSingleTensorMovement{
                  /*parallel_tensor_shape=*/input_shape,
                  /*src_machine_views=*/
                  {
                      BinaryTreePath{{
                          BinaryTreePathEntry::LEFT_CHILD,
                      }},
                  },
                  /*dst_machine_views=*/
                  {
                      BinaryTreePath{{}},
                  },
              },
              AbstractedSingleTensorMovement{
                  /*parallel_tensor_shape=*/input_shape,
                  /*src_machine_views=*/
                  {
                      BinaryTreePath{{
                          BinaryTreePathEntry::RIGHT_CHILD,
                      }},
                  },
                  /*dst_machine_views=*/
                  {
                      BinaryTreePath{{}},
                  },
              },
          }},
          /*pre=*/
          mm_problem_tree_make_parallel(mm_problem_tree_make_leaf(input1_key),
                                        mm_problem_tree_make_leaf(input2_key)),
          /*post=*/mm_problem_tree_make_leaf(ew_op_key));

      CHECK(result == correct);
    }
  }
}
