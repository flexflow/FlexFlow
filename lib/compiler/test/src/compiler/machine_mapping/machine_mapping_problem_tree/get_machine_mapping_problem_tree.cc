#include "compiler/machine_mapping/machine_mapping_problem_tree/get_machine_mapping_problem_tree.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/machine_mapping_problem_tree.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/unmapped_runtime_only_op_cost_estimate_key.dtg.h"
#include "compiler/series_parallel/pcg/get_pcg_balanced_binary_sp_decomposition.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/computation_graph_builder.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "pcg/pcg_from_computation_graph.h"
#include "utils/containers/extend.h"
#include "utils/containers/get_only.h"
#include "utils/containers/require_only_key.h"
#include "utils/full_binary_tree/binary_tree_path.h"
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

    auto mm_problem_tree_make_leaf =
        [](UnmappedRuntimeOnlyOpCostEstimateKey const &k) {
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

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered{
                10_p,
                1_p,
            },
        },
        DataType::FLOAT,
    };
    ParallelTensorShape par_input_shape = lift_shape_to_parallel(input_shape);

    auto make_output_attrs = [](ParallelTensorShape const &shape) {
      return ParallelTensorAttrs{
          /*shape=*/shape,
          /*create_gradients=*/CreateGrad::YES,
      };
    };

    auto make_layer_attrs = [](PCGOperatorAttrs const &op_attrs) {
      return ParallelLayerAttrs{
          /*op_attrs=*/op_attrs,
          /*name=*/std::nullopt,
      };
    };

    PCGOperatorAttrs input_attrs = PCGOperatorAttrs{InputAttrs{input_shape}};

    auto make_input_key =
        [&](ParallelTensorShape const &parallel_tensor_shape) {
          return UnmappedRuntimeOnlyOpCostEstimateKey{
              /*op_attrs=*/input_attrs,
              /*input_shapes=*/{},
              /*weight_shapes=*/{},
              /*output_shapes=*/
              {
                  {
                      TensorSlotName::OUTPUT,
                      parallel_tensor_shape,
                  },
              },
          };
        };

    TaskSpaceCoordinate empty_task_space_coord =
        TaskSpaceCoordinate{OrthotopeCoord{{}}};

    SUBCASE("single layer") {
      ParallelLayerAddedResult input_added =
          add_parallel_layer(pcg,
                             /*layer_attrs=*/make_layer_attrs(input_attrs),
                             /*inputs=*/{},
                             /*output_labels=*/{});
      parallel_layer_guid_t input_layer = input_added.parallel_layer;

      UnmappedRuntimeOnlyOpCostEstimateKey input_key =
          make_input_key(par_input_shape);

      PCGBinarySPDecomposition sp_decomposition =
          PCGBinarySPDecomposition{input_layer};

      MachineMappingProblemTree result =
          get_machine_mapping_problem_tree(pcg, sp_decomposition);
      MachineMappingProblemTree correct = MachineMappingProblemTree{input_key};

      CHECK(result == correct);
    }

    SUBCASE("two layers in series") {
      ParallelLayerAddedResult input_added =
          add_parallel_layer(pcg,
                             /*layer_attrs=*/make_layer_attrs(input_attrs),
                             /*inputs=*/{},
                             /*output_labels=*/{});
      parallel_layer_guid_t input_layer = input_added.parallel_layer;
      parallel_tensor_guid_t input =
          require_only_key(input_added.outputs, TensorSlotName::OUTPUT);

      UnmappedRuntimeOnlyOpCostEstimateKey input_key =
          make_input_key(par_input_shape);

      PCGOperatorAttrs relu_attrs = PCGOperatorAttrs{
          ElementUnaryAttrs{
              /*op_type=*/ElementUnaryOp::RELU,
              /*scalar=*/std::nullopt,
          },
      };
      ParallelTensorShape relu_output_shape = par_input_shape;
      ParallelLayerAddedResult relu_added = add_parallel_layer(
          /*pcg=*/pcg,
          /*layer_attrs=*/make_layer_attrs(relu_attrs),
          /*inputs=*/
          {
              {
                  TensorSlotName::INPUT,
                  input,
              },
          },
          /*weights=*/{});
      parallel_layer_guid_t relu_layer = relu_added.parallel_layer;
      parallel_tensor_guid_t relu_output =
          require_only_key(relu_added.outputs, TensorSlotName::OUTPUT);

      UnmappedRuntimeOnlyOpCostEstimateKey relu_key =
          UnmappedRuntimeOnlyOpCostEstimateKey{
              /*op_attrs=*/relu_attrs,
              /*input_shapes=*/
              {
                  {
                      TensorSlotName::INPUT,
                      par_input_shape,
                  },
              },
              /*weight_shapes=*/{},
              /*output_shapes=*/
              {
                  {
                      TensorSlotName::OUTPUT,
                      relu_output_shape,
                  },
              },
          };

      PCGBinarySPDecomposition sp_decomposition = pcg_make_series(
          pcg_make_leaf(input_layer), pcg_make_leaf(relu_layer));

      MachineMappingProblemTree result =
          get_machine_mapping_problem_tree(pcg, sp_decomposition);

      MachineMappingProblemTree correct = mm_problem_tree_make_series(
          AbstractedTensorSetMovement{
              /*single_tensor_movements=*/{AbstractedSingleTensorMovement{
                  /*src_op_tree_path=*/binary_tree_root_path(),
                  /*edge_to_size=*/
                  {
                      {
                          AbstractedSingleTensorCommunicationEdge{
                              /*src_coord=*/empty_task_space_coord,
                              /*dst=*/
                              AbstractedDevice{
                                  /*operator_tree_path=*/
                                  binary_tree_root_path(),
                                  /*task_space_coordinate=*/
                                  empty_task_space_coord,
                              },
                          },
                          get_piece_size_in_bytes(par_input_shape),
                      },
                  },
              }}},
          mm_problem_tree_make_leaf(input_key),
          mm_problem_tree_make_leaf(relu_key));

      CHECK(result == correct);
    }

    SUBCASE("two layers in parallel") {
      ParallelLayerAddedResult input1_added =
          pcg_add_input_layer(pcg, input_shape);
      parallel_layer_guid_t input1_layer = input1_added.parallel_layer;
      UnmappedRuntimeOnlyOpCostEstimateKey input1_key =
          make_input_key(par_input_shape);

      ParallelLayerAddedResult input2_added =
          pcg_add_input_layer(pcg, input_shape);
      parallel_layer_guid_t input2_layer = input2_added.parallel_layer;
      UnmappedRuntimeOnlyOpCostEstimateKey input2_key =
          make_input_key(par_input_shape);

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
      parallel_tensor_guid_t input1_tensor =
          require_only_key(input1_added.outputs, TensorSlotName::OUTPUT);
      UnmappedRuntimeOnlyOpCostEstimateKey input1_key =
          make_input_key(par_input_shape);

      ParallelLayerAddedResult input2_added =
          pcg_add_input_layer(pcg, input_shape);
      parallel_layer_guid_t input2_layer = input2_added.parallel_layer;
      parallel_tensor_guid_t input2_tensor =
          require_only_key(input2_added.outputs, TensorSlotName::OUTPUT);
      UnmappedRuntimeOnlyOpCostEstimateKey input2_key =
          make_input_key(par_input_shape);

      PCGOperatorAttrs ew_op_attrs = PCGOperatorAttrs{
          ElementBinaryAttrs{
              /*op=*/ElementBinaryOp::ADD,
              /*compute_type=*/DataType::FLOAT,
              /*should_broadcast_lhs=*/false,
              /*should_broadcast_rhs=*/false,
          },
      };
      ParallelTensorShape ew_op_output_shape = par_input_shape;
      ParallelLayerAddedResult ew_op_added = add_parallel_layer(
          /*pcg=*/pcg,
          /*layer_attrs=*/make_layer_attrs(ew_op_attrs),
          /*inputs=*/
          {
              {
                  TensorSlotName::LHS_INPUT,
                  input1_tensor,
              },
              {
                  TensorSlotName::RHS_INPUT,
                  input2_tensor,
              },
          },
          /*outputs=*/{});
      parallel_layer_guid_t ew_op_layer = ew_op_added.parallel_layer;

      UnmappedRuntimeOnlyOpCostEstimateKey ew_op_key =
          UnmappedRuntimeOnlyOpCostEstimateKey{
              /*op_attrs=*/ew_op_attrs,
              /*input_shapes=*/
              {
                  {
                      TensorSlotName::LHS_INPUT,
                      par_input_shape,
                  },
                  {
                      TensorSlotName::RHS_INPUT,
                      par_input_shape,
                  },
              },
              /*weight_shapes=*/{},
              /*output_shapes=*/
              {
                  {
                      TensorSlotName::OUTPUT,
                      ew_op_output_shape,
                  },
              },
          };

      PCGBinarySPDecomposition sp_decomposition =
          pcg_make_series(pcg_make_parallel(pcg_make_leaf(input1_layer),
                                            pcg_make_leaf(input2_layer)),
                          pcg_make_leaf(ew_op_layer));

      MachineMappingProblemTree result =
          get_machine_mapping_problem_tree(pcg, sp_decomposition);

      BinaryTreePath src1_path = BinaryTreePath{{
          BinaryTreePathEntry::LEFT_CHILD,
      }};

      BinaryTreePath src2_path = BinaryTreePath{{
          BinaryTreePathEntry::RIGHT_CHILD,
      }};

      AbstractedSingleTensorCommunicationEdge edge =
          AbstractedSingleTensorCommunicationEdge{
              /*src_coord=*/empty_task_space_coord,
              /*dst=*/
              AbstractedDevice{
                  /*operator_tree_path=*/binary_tree_root_path(),
                  /*task_space_coordinate=*/empty_task_space_coord,
              },
          };

      MachineMappingProblemTree correct = mm_problem_tree_make_series(
          AbstractedTensorSetMovement{
              /*single_tensor_movements=*/{
                  AbstractedSingleTensorMovement{
                      /*src_op_tree_path=*/src1_path,
                      /*edge_to_size=*/
                      {
                          {edge, get_piece_size_in_bytes(par_input_shape)},
                      },
                  },
                  AbstractedSingleTensorMovement{
                      /*src_op_tree_path=*/src2_path,
                      /*edge_to_size=*/
                      {
                          {edge, get_piece_size_in_bytes(par_input_shape)},
                      },
                  },
              },
          },
          /*pre=*/
          mm_problem_tree_make_parallel(mm_problem_tree_make_leaf(input1_key),
                                        mm_problem_tree_make_leaf(input2_key)),
          /*post=*/mm_problem_tree_make_leaf(ew_op_key));

      CHECK(result == correct);
    }
  }

  TEST_CASE("from pcg") {
    ComputationGraph cg = [&] {
      ComputationGraphBuilder b;
      TensorShape input_tensor_shape = TensorShape{
          TensorDims{
              FFOrdered<positive_int>{
                  32_p,
                  64_p,
              },
          },
          DataType::FLOAT,
      };
      tensor_guid_t t = b.create_input(input_tensor_shape, CreateGrad::YES);
      t = b.dense(t,
                  /*outDim=*/16_p,
                  /*activation=*/std::nullopt);
      t = b.gelu(t);
      t = b.dense(t,
                  /*outDim=*/12_p,
                  /*activation=*/std::nullopt,
                  /*use_bias=*/false,
                  /*data_type=*/DataType::FLOAT,
                  /*kernel_initializer=*/std::nullopt,
                  /*bias_initializer=*/std::nullopt);
      t = b.relu(t);
      t = b.dense(t,
                  /*outDim=*/8_p,
                  /*activation=*/Activation::RELU);
      return b.computation_graph;
    }();

    ParallelComputationGraph pcg = pcg_from_computation_graph(cg);

    PCGBinarySPDecomposition sp_decomp =
        expect(get_pcg_balanced_binary_sp_decomposition(pcg),
               "Failed to get SP decomposition of PCG");

    MachineMappingProblemTree problem_tree =
        get_machine_mapping_problem_tree(pcg, sp_decomp);
  }
}
