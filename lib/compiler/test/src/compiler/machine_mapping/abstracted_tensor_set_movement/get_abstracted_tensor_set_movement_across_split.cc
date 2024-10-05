#include "compiler/machine_mapping/abstracted_tensor_set_movement/get_abstracted_tensor_set_movement_across_split.h"
#include "compiler/machine_mapping/transitive_reduced_pcg.h"
#include "compiler/series_parallel/pcg_binary_sp_decomposition.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "utils/containers/get_only.h"
#include "utils/full_binary_tree/binary_tree_path.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_abstracted_tensor_set_movement_across_split") {
    ParallelComputationGraph pcg = empty_parallel_computation_graph();

    ParallelTensorShape input_shape = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{10, 2},
                ShardParallelDim{12, 1},
            },
            ReplicaParallelDimSet{
                SumDegree{1},
                DiscardCopyDegree{1},
            },
        },
        DataType::FLOAT,
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
                /*type=*/OperatorType::EW_ADD,
                /*compute_type=*/DataType::FLOAT,
                /*should_broadcast_lhs=*/false,
                /*should_broadcast_rhs=*/false,
            },
        },
        /*name=*/std::nullopt,
    };

    ParallelTensorAttrs relu_output_attrs = ParallelTensorAttrs{
        /*shape=*/input_shape,
        /*sync_type=*/std::nullopt,
        /*initializer=*/std::nullopt,
        /*create_gradients=*/CreateGrad::YES,
    };

    SUBCASE("no edges across split") {
      ParallelLayerAddedResult input1 = pcg_add_input_layer(pcg, input_shape);
      ParallelLayerAddedResult input2 = pcg_add_input_layer(pcg, input_shape);

      PCGBinarySeriesSplit split = require_series(
          make_pcg_series_split(make_pcg_leaf_node(input1.parallel_layer),
                                make_pcg_leaf_node(input2.parallel_layer)));

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

      ParallelLayerAddedResult layer_1 = add_parallel_layer(
          pcg, relu_attrs, {get_only(input.outputs)}, {relu_output_attrs});
      ParallelLayerAddedResult layer_2 = add_parallel_layer(
          pcg, relu_attrs, {get_only(layer_1.outputs)}, {relu_output_attrs});

      PCGBinarySeriesSplit split = require_series(make_pcg_series_split(
          make_pcg_series_split(make_pcg_leaf_node(input.parallel_layer),
                                make_pcg_leaf_node(layer_1.parallel_layer)),
          make_pcg_leaf_node(layer_2.parallel_layer)));

      AbstractedTensorSetMovement result =
          get_abstracted_tensor_set_movement_across_split(
              pcg_get_transitive_reduction(pcg), split);

      AbstractedTensorSetMovement correct = AbstractedTensorSetMovement{
          /*single_tensor_movements=*/{
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
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("does not include edges removed by transitive reduction") {
      ParallelLayerAddedResult input = pcg_add_input_layer(pcg, input_shape);

      ParallelLayerAddedResult layer_1 = add_parallel_layer(
          pcg, relu_attrs, {get_only(input.outputs)}, {relu_output_attrs});

      ParallelLayerAddedResult layer_2 = add_parallel_layer(
          pcg, relu_attrs, {get_only(layer_1.outputs)}, {relu_output_attrs});

      ParallelLayerAddedResult layer_3 = add_parallel_layer(
          pcg,
          ew_add_attrs,
          {get_only(layer_1.outputs), get_only(layer_2.outputs)},
          {relu_output_attrs});

      PCGBinarySeriesSplit split = require_series(make_pcg_series_split(
          make_pcg_series_split(
              make_pcg_leaf_node(input.parallel_layer),
              make_pcg_series_split(
                  make_pcg_leaf_node(layer_1.parallel_layer),
                  make_pcg_leaf_node(layer_2.parallel_layer))),
          make_pcg_leaf_node(layer_3.parallel_layer)));

      AbstractedTensorSetMovement result =
          get_abstracted_tensor_set_movement_across_split(
              pcg_get_transitive_reduction(pcg), split);

      AbstractedTensorSetMovement correct = AbstractedTensorSetMovement{
          /*single_tensor_movements=*/{
              AbstractedSingleTensorMovement{
                  /*parallel_tensor_shape=*/input_shape,
                  /*src_machine_views=*/
                  {
                      BinaryTreePath{{
                          BinaryTreePathEntry::RIGHT_CHILD,
                          BinaryTreePathEntry::RIGHT_CHILD,
                      }},
                  },
                  /*dst_machine_views=*/
                  {
                      BinaryTreePath{{}},
                  },
              },
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("single tensor, multiple consumers across split") {
      ParallelLayerAddedResult input = pcg_add_input_layer(pcg, input_shape);

      ParallelLayerAddedResult layer_1 = add_parallel_layer(
          pcg, relu_attrs, {get_only(input.outputs)}, {relu_output_attrs});

      ParallelLayerAddedResult layer_2 = add_parallel_layer(
          pcg, relu_attrs, {get_only(layer_1.outputs)}, {relu_output_attrs});

      ParallelLayerAddedResult layer_3 = add_parallel_layer(
          pcg, relu_attrs, {get_only(layer_1.outputs)}, {relu_output_attrs});

      PCGBinarySeriesSplit split = require_series(make_pcg_series_split(
          make_pcg_series_split(make_pcg_leaf_node(input.parallel_layer),
                                make_pcg_leaf_node(layer_1.parallel_layer)),
          make_pcg_parallel_split(make_pcg_leaf_node(layer_2.parallel_layer),
                                  make_pcg_leaf_node(layer_3.parallel_layer))));

      AbstractedTensorSetMovement result =
          get_abstracted_tensor_set_movement_across_split(
              pcg_get_transitive_reduction(pcg), split);

      AbstractedTensorSetMovement correct = AbstractedTensorSetMovement{
          /*single_tensor_movements=*/{
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
                      BinaryTreePath{{
                          BinaryTreePathEntry::LEFT_CHILD,
                      }},
                      BinaryTreePath{{
                          BinaryTreePathEntry::RIGHT_CHILD,
                      }},
                  },
              },
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("multiple tensors, multiple consumers across split") {
      ParallelLayerAddedResult input = pcg_add_input_layer(pcg, input_shape);

      ParallelLayerAddedResult layer_1 = add_parallel_layer(
          pcg, relu_attrs, {get_only(input.outputs)}, {relu_output_attrs});

      ParallelLayerAddedResult layer_2 = add_parallel_layer(
          pcg, relu_attrs, {get_only(input.outputs)}, {relu_output_attrs});

      ParallelLayerAddedResult layer_3 = add_parallel_layer(
          pcg, relu_attrs, {get_only(layer_1.outputs)}, {relu_output_attrs});

      ParallelLayerAddedResult layer_4 = add_parallel_layer(
          pcg,
          ew_add_attrs,
          {get_only(layer_1.outputs), get_only(layer_2.outputs)},
          {relu_output_attrs});

      PCGBinarySeriesSplit split = require_series(make_pcg_series_split(
          make_pcg_series_split(
              make_pcg_leaf_node(input.parallel_layer),
              make_pcg_parallel_split(
                  make_pcg_leaf_node(layer_1.parallel_layer),
                  make_pcg_leaf_node(layer_2.parallel_layer))),
          make_pcg_parallel_split(make_pcg_leaf_node(layer_3.parallel_layer),
                                  make_pcg_leaf_node(layer_4.parallel_layer))));

      AbstractedTensorSetMovement result =
          get_abstracted_tensor_set_movement_across_split(
              pcg_get_transitive_reduction(pcg), split);

      AbstractedTensorSetMovement correct = AbstractedTensorSetMovement{
          /*single_tensor_movements=*/{
              AbstractedSingleTensorMovement{
                  /*parallel_tensor_shape=*/input_shape,
                  /*src_machine_views=*/
                  {
                      BinaryTreePath{{
                          BinaryTreePathEntry::RIGHT_CHILD,
                          BinaryTreePathEntry::LEFT_CHILD,
                      }},
                  },
                  /*dst_machine_views=*/
                  {
                      BinaryTreePath{{
                          BinaryTreePathEntry::LEFT_CHILD,
                      }},
                      BinaryTreePath{{
                          BinaryTreePathEntry::RIGHT_CHILD,
                      }},
                  },
              },
              AbstractedSingleTensorMovement{
                  /*parallel_tensor_shape=*/input_shape,
                  /*src_machine_views=*/
                  {
                      BinaryTreePath{{
                          BinaryTreePathEntry::RIGHT_CHILD,
                          BinaryTreePathEntry::RIGHT_CHILD,
                      }},
                  },
                  /*dst_machine_views=*/
                  {
                      BinaryTreePath{{
                          BinaryTreePathEntry::RIGHT_CHILD,
                      }},
                  },
              },
          },
      };

      CHECK(result == correct);
    }
  }
}
