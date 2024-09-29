#include "compiler/machine_mapping/get_machine_mapping_problem_tree.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree.h"
#include "compiler/series_parallel/pcg_binary_sp_decomposition.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "utils/containers/get_only.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_machine_mapping_problem_tree") {
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

    SUBCASE("single layer") {
      ParallelLayerAddedResult input_added = pcg_add_input_layer(pcg, input_shape);
      parallel_layer_guid_t input_layer = input_added.parallel_layer;

      PCGBinarySPDecomposition sp_decomposition = \
          make_pcg_leaf_node(input_layer);

      MachineMappingProblemTree result = get_machine_mapping_problem_tree(pcg, sp_decomposition);
      MachineMappingProblemTree correct = mm_problem_tree_make_leaf(input_attrs);

      CHECK(result == correct);
    }

    SUBCASE("two layers in series") {
      ParallelLayerAddedResult input_added = pcg_add_input_layer(pcg, input_shape);
      parallel_layer_guid_t input_layer = input_added.parallel_layer;
      parallel_tensor_guid_t input = get_only(input_added.outputs);

      PCGOperatorAttrs relu_attrs = PCGOperatorAttrs{
        ElementUnaryAttrs{
          /*op_type=*/OperatorType::RELU,
          /*scalar=*/std::nullopt,
        },
      };
      ParallelTensorShape relu_output_shape = input_shape;
      ParallelLayerAddedResult relu_added = add_parallel_layer(pcg,
                                                                make_layer_attrs(relu_attrs),
                                                                {input},
                                                                {make_output_attrs(relu_output_shape)});
      parallel_layer_guid_t relu_layer = relu_added.parallel_layer;
      parallel_tensor_guid_t relu_output = get_only(relu_added.outputs);

      PCGBinarySPDecomposition sp_decomposition = \
        make_pcg_series_split(
          make_pcg_leaf_node(input_layer),
          make_pcg_leaf_node(relu_layer));

      MachineMappingProblemTree result = get_machine_mapping_problem_tree(pcg, sp_decomposition);

      MachineMappingProblemTree correct = \
        mm_problem_tree_make_series_split(
          AbstractedTensorSetMovement{{
            AbstractedSingleTensorMovement{
              input_shape,
              {input_layer},
              {relu_layer},
            },
          }},
          mm_problem_tree_make_leaf(input_attrs),
          mm_problem_tree_make_leaf(relu_attrs));

      CHECK(result == correct);
    }

    SUBCASE("two layers in parallel") {
      ParallelLayerAddedResult input1_added = pcg_add_input_layer(pcg, input_shape);
      parallel_layer_guid_t input1_layer = input1_added.parallel_layer;

      ParallelLayerAddedResult input2_added = pcg_add_input_layer(pcg, input_shape);
      parallel_layer_guid_t input2_layer = input2_added.parallel_layer;

      PCGBinarySPDecomposition sp_decomposition = \
        make_pcg_series_split(
          make_pcg_leaf_node(input1_layer),
          make_pcg_leaf_node(input2_layer));

      MachineMappingProblemTree result = get_machine_mapping_problem_tree(pcg, sp_decomposition);

      MachineMappingProblemTree correct = \
        mm_problem_tree_make_parallel_split(
          mm_problem_tree_make_leaf(input_attrs),
          mm_problem_tree_make_leaf(input_attrs));

      CHECK(result == correct);
    }

    SUBCASE("multiple tensors across split") {
      ParallelLayerAddedResult input1_added = pcg_add_input_layer(pcg, input_shape);
      parallel_layer_guid_t input1_layer = input1_added.parallel_layer;
      parallel_tensor_guid_t input1_tensor = get_only(input1_added.outputs);

      ParallelLayerAddedResult input2_added = pcg_add_input_layer(pcg, input_shape);
      parallel_layer_guid_t input2_layer = input2_added.parallel_layer;
      parallel_tensor_guid_t input2_tensor = get_only(input2_added.outputs);

      PCGOperatorAttrs ew_op_attrs = PCGOperatorAttrs{
        ElementBinaryAttrs{
          /*type=*/OperatorType::EW_ADD,
          /*compute_type=*/DataType::FLOAT,
          /*should_broadcast_lhs=*/false,
          /*should_broadcast_rhs=*/false,
        },
      };
      ParallelTensorShape ew_op_output_shape = input_shape;
      ParallelLayerAddedResult ew_op_added = add_parallel_layer(pcg,
                                                                make_layer_attrs(ew_op_attrs),
                                                                {input1_tensor, input2_tensor},
                                                                {make_output_attrs(ew_op_output_shape)});
      parallel_layer_guid_t ew_op_layer = ew_op_added.parallel_layer;

      PCGBinarySPDecomposition sp_decomposition = \
        make_pcg_series_split(
          make_pcg_parallel_split(
            make_pcg_leaf_node(input1_layer),
            make_pcg_leaf_node(input2_layer)),
          make_pcg_leaf_node(ew_op_layer));

      MachineMappingProblemTree result = get_machine_mapping_problem_tree(pcg, sp_decomposition);

      MachineMappingProblemTree correct = \
        mm_problem_tree_make_series_split(
          AbstractedTensorSetMovement{{
            AbstractedSingleTensorMovement{
              /*parallel_tensor_shape=*/input_shape,
              /*src_machine_views=*/{input1_layer},
              /*dst_machine_views=*/{ew_op_layer},
            },
            AbstractedSingleTensorMovement{
              /*parallel_tensor_shape=*/input_shape,
              /*src_machine_views=*/{input2_layer},
              /*dst_machine_views=*/{ew_op_layer},
            },
          }},
          /*pre=*/mm_problem_tree_make_parallel_split(
                    mm_problem_tree_make_leaf(input_attrs),
                    mm_problem_tree_make_leaf(input_attrs)),
          /*post=*/mm_problem_tree_make_leaf(ew_op_attrs));

      CHECK(result == correct);
    }
  }
}
