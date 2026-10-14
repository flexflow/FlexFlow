#include "compiler/machine_mapping/get_optimal_machine_mapping.h"
#include "compiler/cost_estimator/runtime_only_op_cost_estimate_key.dtg.h"
#include "compiler/cost_estimator/runtime_only_op_cost_metrics.dtg.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_tensor_set_movement.h"
#include "compiler/machine_mapping/allowed_machine_views.h"
#include "compiler/machine_mapping/machine_mapping_cache.h"
#include "compiler/machine_mapping/machine_mapping_constraints.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/machine_mapping_problem_tree.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/unmapped_op_cost_estimate_key.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/unmapped_runtime_only_op_cost_estimate_key.h"
#include "compiler/machine_mapping/machine_view.h"
#include "internal/runtime_only_cost_estimator_for_test.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/task_space_coordinate.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "utils/containers/get_only.h"
#include "utils/full_binary_tree/binary_tree_path.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_optimal_machine_mapping") {
    auto make_leaf = [](UnmappedRuntimeOnlyOpCostEstimateKey const &k) {
      return MachineMappingProblemTree{k};
    };

    auto make_series_split =
        [](AbstractedTensorSetMovement const &tensor_set_movement,
           MachineMappingProblemTree const &lhs,
           MachineMappingProblemTree const &rhs) {
          return MachineMappingProblemTree{
              MMProblemTreeSeriesSplit{
                  /*tensor_set_movement=*/tensor_set_movement,
                  /*left_child=*/lhs,
                  /*right_child=*/rhs,
              },
          };
        };

    auto make_parallel_split = [](MachineMappingProblemTree const &lhs,
                                  MachineMappingProblemTree const &rhs) {
      return MachineMappingProblemTree{
          MMProblemTreeParallelSplit{
              /*left_child=*/lhs,
              /*right_child=*/rhs,
          },
      };
    };

    MachineView mv_stride_1 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/0_n,
            /*device_idx=*/0_n,
        },
        /*dimensions=*/
        {
            MachineViewDimension{
                stride_t{1_p},
                MachineSpecificationDimension::INTER_NODE,
            },
        },
    };

    MachineView mv_stride_2 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/0_n,
            /*device_idx=*/0_n,
        },
        /*dimensions=*/
        {
            MachineViewDimension{
                stride_t{2_p},
                MachineSpecificationDimension::INTER_NODE,
            },
        },
    };

    MachineComputeResourceSlice four_nodes_resources =
        MachineComputeResourceSlice{
            /*num_nodes=*/4_p,
            /*num_gpus_per_node=*/1_p,
        };

    MachineComputeResourceSlice three_nodes_resources =
        MachineComputeResourceSlice{
            /*num_nodes=*/3_p,
            /*num_gpus_per_node=*/1_p,
        };

    MachineComputeResourceSlice two_nodes_resources =
        MachineComputeResourceSlice{
            /*num_nodes=*/2_p,
            /*num_gpus_per_node=*/1_p,
        };

    MachineComputeResourceSlice one_node_resources =
        MachineComputeResourceSlice{
            /*num_nodes=*/1_p,
            /*num_gpus_per_node=*/1_p,
        };

    TensorShape tensor_shape = TensorShape{
        TensorDims{
            FFOrdered{
                10_p,
                8_p,
            },
        },
        DataType::FLOAT,
    };

    BinaryTreePath src_path = binary_tree_root_path();

    ParallelLayerGuidObliviousMachineMapping mm1 =
        ParallelLayerGuidObliviousMachineMapping{{
            {binary_tree_root_path(), mv_stride_1},
        }};
    ParallelLayerGuidObliviousMachineMapping mm2 =
        ParallelLayerGuidObliviousMachineMapping{{
            {binary_tree_root_path(), mv_stride_2},
        }};

    OperatorTaskSpace task_space = OperatorTaskSpace{
        MinimalOrthotope{{
            2_ge2,
        }},
    };

    MachineMappingCache cache = empty_machine_mapping_cache();

    ParallelTensorShape par_tensor_shape = lift_to_parallel_with_degrees(
        tensor_shape,
        ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/
            FFOrdered<positive_int>{
                2_p,
                1_p,
            },
        });

    UnmappedRuntimeOnlyOpCostEstimateKey k1 =
        UnmappedRuntimeOnlyOpCostEstimateKey{
            /*op_attrs=*/PCGOperatorAttrs{
                ElementUnaryAttrs{
                    /*type=*/ElementUnaryOp::GELU,
                    /*scalar=*/std::nullopt,
                },
            },
            /*input_shapes=*/
            {
                {
                    TensorSlotName::INPUT,
                    par_tensor_shape,
                },
            },
            /*weight_shapes=*/{},
            /*output_shapes=*/
            {
                {
                    TensorSlotName::OUTPUT,
                    par_tensor_shape,
                },
            },
        };

    UnmappedRuntimeOnlyOpCostEstimateKey k2 =
        UnmappedRuntimeOnlyOpCostEstimateKey{
            /*op_attrs=*/PCGOperatorAttrs{
                ElementUnaryAttrs{
                    /*type=*/ElementUnaryOp::RELU,
                    /*scalar=*/std::nullopt,
                },
            },
            /*input_shapes=*/
            {
                {
                    TensorSlotName::INPUT,
                    par_tensor_shape,
                },
            },
            /*weight_shapes=*/{},
            /*output_shapes=*/
            {
                {
                    TensorSlotName::OUTPUT,
                    par_tensor_shape,
                },
            },
        };

    auto mk_cost_metrics = [&](float total_cost) {
      return RuntimeOnlyOpCostMetrics{
          /*forward_runtime=*/milliseconds_t{total_cost / 2},
          /*backward_runtime=*/milliseconds_t{total_cost / 2},
      };
    };

    auto mk_cost_entry = [&](UnmappedRuntimeOnlyOpCostEstimateKey const &key,
                             MachineView const &mv,
                             float total_cost) {
      return std::pair{
          map_unmapped_runtime_only_op_cost_estimate_key(key, mv),
          mk_cost_metrics(total_cost),
      };
    };

    SUBCASE("single layer") {
      MachineMappingProblemTree problem_tree = make_leaf(k1);

      MachineMappingConstraints constraints =
          get_unconstrained_solution_for_layers(
              get_all_leaf_paths(problem_tree));

      auto allowed_machine_views =
          [&](UnmappedRuntimeOnlyOpCostEstimateKey const &k,
              MachineComputeResourceSlice const &resources) {
            ASSERT(k == k1);
            ASSERT(resources == four_nodes_resources);

            return std::set<MachineView>{
                mv_stride_1,
                mv_stride_2,
            };
          };

      RuntimeOnlyCostEstimator runtime_only_cost_estimator =
          make_fake_runtime_only_cost_estimator(
              {
                  mk_cost_entry(k1, mv_stride_1, 1),
                  mk_cost_entry(k1, mv_stride_2, 2),
              },
              std::map<TensorSetMovement, milliseconds_t>{{}});

      MachineMappingContext context = MachineMappingContext{
          /*cost_estimator=*/runtime_only_cost_estimator,
          /*allowed_machine_views=*/allowed_machine_views,
      };

      MachineMappingResult result = get_optimal_machine_mapping(
          cache, context, problem_tree, four_nodes_resources, constraints);

      MachineMappingResult correct = MachineMappingResult{
          FeasibleMachineMappingResult{
              /*runtime=*/1_ms,
              /*machine_mapping=*/
              ParallelLayerGuidObliviousMachineMapping{{
                  {binary_tree_root_path(), mv_stride_1},
              }},
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("pair of layers in sequence") {
      AbstractedTensorSetMovement k1_to_k2 = AbstractedTensorSetMovement{
          /*single_tensor_movements=*/{
              AbstractedSingleTensorMovement{
                  /*src_op_tree_path=*/binary_tree_root_path(),
                  /*edge_to_size=*/
                  {
                      {
                          AbstractedSingleTensorCommunicationEdge{
                              /*src_coord=*/make_task_space_coordinate({0_n}),
                              /*dst=*/
                              AbstractedDevice{
                                  /*operator_tree_path=*/
                                  binary_tree_root_path(),
                                  /*task_space_coordinate=*/
                                  make_task_space_coordinate({0_n}),
                              },
                          },
                          get_size_in_bytes(tensor_shape),
                      },
                      {
                          AbstractedSingleTensorCommunicationEdge{
                              /*src_coord=*/make_task_space_coordinate({1_n}),
                              /*dst=*/
                              AbstractedDevice{
                                  /*operator_tree_path=*/
                                  binary_tree_root_path(),
                                  /*task_space_coordinate=*/
                                  make_task_space_coordinate({1_n}),
                              },
                          },
                          get_size_in_bytes(tensor_shape),
                      },
                  },
              },
          },
      };

      MachineMappingProblemTree problem_tree =
          make_series_split(k1_to_k2, make_leaf(k1), make_leaf(k2));

      auto mk_tensor_set_movement = [&](MachineView const &src_mv,
                                        MachineView const &dst_mv) {
        MachineSpaceStencil src_stencil = MachineSpaceStencil{
            /*operator_task_space=*/task_space,
            /*machine_view=*/src_mv,
        };

        MachineSpaceStencil dst_stencil = MachineSpaceStencil{
            /*operator_task_space=*/task_space,
            /*machine_view=*/dst_mv,
        };

        return concretize_abstracted_tensor_set_movement(
            k1_to_k2,
            /*pre_machine_stencils=*/{{binary_tree_root_path(), src_stencil}},
            /*post_machine_stencils=*/{{binary_tree_root_path(), dst_stencil}});
      };

      auto allowed_machine_views =
          [&](UnmappedRuntimeOnlyOpCostEstimateKey const &k,
              MachineComputeResourceSlice const &resources)
          -> std::set<MachineView> {
        std::set<MachineView> result;

        if (resources == four_nodes_resources) {
          result = std::set<MachineView>{mv_stride_1, mv_stride_2};
        } else if (resources == three_nodes_resources) {
          result = std::set<MachineView>{mv_stride_1, mv_stride_2};
        } else if (resources == two_nodes_resources) {
          result = std::set<MachineView>{mv_stride_1};
        } else {
          result = std::set<MachineView>{};
        }

        for (MachineView const &mv : result) {
          OperatorTaskSpace op_task_space =
              get_operator_task_space_for_runtime_only_op_cost_estimate_key(k);
          ASSERT(is_valid_machine_view(mv, op_task_space, resources));
        }

        return result;
      };

      MachineMappingConstraints constraints =
          get_unconstrained_solution_for_layers(
              get_all_leaf_paths(problem_tree));

      SUBCASE("solution requires taking comm cost into account") {
        RuntimeOnlyCostEstimator runtime_only_cost_estimator =
            make_fake_runtime_only_cost_estimator(
                std::map<RuntimeOnlyOpCostEstimateKey,
                         RuntimeOnlyOpCostMetrics>{{
                    mk_cost_entry(k1, mv_stride_1, 1),
                    mk_cost_entry(k1, mv_stride_2, 3),
                    mk_cost_entry(k2, mv_stride_1, 4),
                    mk_cost_entry(k2, mv_stride_2, 1),
                }},
                std::map<TensorSetMovement, milliseconds_t>{{
                    {
                        TensorSetMovement{{}},
                        0.0_ms,
                    },
                    {
                        mk_tensor_set_movement(mv_stride_1, mv_stride_2),
                        5_ms,
                    },
                    {
                        mk_tensor_set_movement(mv_stride_2, mv_stride_1),
                        5_ms,
                    },
                }});

        MachineMappingContext context = MachineMappingContext{
            /*cost_estimator=*/runtime_only_cost_estimator,
            /*allowed_machine_views=*/allowed_machine_views,
        };

        MachineMappingResult result = get_optimal_machine_mapping(
            cache, context, problem_tree, four_nodes_resources, constraints);

        MachineMappingResult correct = MachineMappingResult{
            FeasibleMachineMappingResult{
                /*runtime=*/1.0_ms + 3.0_ms,
                /*machine_mapping=*/
                ParallelLayerGuidObliviousMachineMapping{{
                    {
                        BinaryTreePath{{
                            BinaryTreePathEntry::LEFT_CHILD,
                        }},
                        mv_stride_2,
                    },
                    {
                        BinaryTreePath{{
                            BinaryTreePathEntry::RIGHT_CHILD,
                        }},
                        mv_stride_2,
                    },
                }},
            },
        };

        CHECK(result == correct);
      }

      SUBCASE("solution places operators on different machine views") {
        RuntimeOnlyCostEstimator runtime_only_cost_estimator =
            make_fake_runtime_only_cost_estimator(
                std::map<RuntimeOnlyOpCostEstimateKey,
                         RuntimeOnlyOpCostMetrics>{{
                    mk_cost_entry(k1, mv_stride_1, 1),
                    mk_cost_entry(k1, mv_stride_2, 3),
                    mk_cost_entry(k2, mv_stride_1, 4),
                    mk_cost_entry(k2, mv_stride_2, 1),
                }},
                std::map<TensorSetMovement, milliseconds_t>{{
                    {
                        TensorSetMovement{{}},
                        0.0_ms,
                    },
                    {
                        mk_tensor_set_movement(mv_stride_1, mv_stride_2),
                        1_ms,
                    },
                    {
                        mk_tensor_set_movement(mv_stride_2, mv_stride_1),
                        1_ms,
                    },
                }});

        MachineMappingContext context = MachineMappingContext{
            /*cost_estimator=*/runtime_only_cost_estimator,
            /*allowed_machine_views=*/allowed_machine_views,
        };

        MachineMappingResult result = get_optimal_machine_mapping(
            cache, context, problem_tree, four_nodes_resources, constraints);

        MachineMappingResult correct = MachineMappingResult{
            FeasibleMachineMappingResult{
                /*runtime=*/1.0_ms + 1.0_ms + 1.0_ms,
                /*machine_mapping=*/
                ParallelLayerGuidObliviousMachineMapping{{
                    {
                        BinaryTreePath{{
                            BinaryTreePathEntry::LEFT_CHILD,
                        }},
                        mv_stride_1,
                    },
                    {
                        BinaryTreePath{{
                            BinaryTreePathEntry::RIGHT_CHILD,
                        }},
                        mv_stride_2,
                    },
                }},
            },
        };

        CHECK(result == correct);
      }
    }

    SUBCASE("pair of layers in parallel") {
      MachineMappingProblemTree problem_tree =
          make_parallel_split(make_leaf(k1), make_leaf(k2));

      MachineMappingConstraints constraints =
          get_unconstrained_solution_for_layers(
              get_all_leaf_paths(problem_tree));

      auto allowed_machine_views =
          [&](UnmappedRuntimeOnlyOpCostEstimateKey const &k,
              MachineComputeResourceSlice const &resources) {
            if (resources == four_nodes_resources) {
              return std::set<MachineView>{mv_stride_1, mv_stride_2};
            } else if (resources == three_nodes_resources) {
              return std::set<MachineView>{mv_stride_1, mv_stride_2};
            } else if (resources == two_nodes_resources) {
              return std::set<MachineView>{mv_stride_1};
            } else {
              return std::set<MachineView>{};
            }
          };

      SUBCASE("cannot use overlapping machine views in parallel") {
        RuntimeOnlyCostEstimator runtime_only_cost_estimator =
            make_fake_runtime_only_cost_estimator(
                std::map<RuntimeOnlyOpCostEstimateKey,
                         RuntimeOnlyOpCostMetrics>{{
                    mk_cost_entry(k1, mv_stride_1, 1),
                    mk_cost_entry(k1, mv_stride_2, 3),
                    mk_cost_entry(k2, mv_stride_1, 4),
                    mk_cost_entry(k2, mv_stride_2, 1),
                }},
                std::map<TensorSetMovement, milliseconds_t>{{
                    {
                        TensorSetMovement{{}},
                        0.0_ms,
                    },
                }});

        MachineMappingContext context = MachineMappingContext{
            /*cost_estimator=*/runtime_only_cost_estimator,
            /*allowed_machine_views=*/allowed_machine_views,
        };

        MachineMappingResult result = get_optimal_machine_mapping(
            cache, context, problem_tree, four_nodes_resources, constraints);

        MachineMappingResult correct = MachineMappingResult{
            FeasibleMachineMappingResult{
                /*runtime=*/2_ms,
                /*machine_mapping=*/
                ParallelLayerGuidObliviousMachineMapping{{
                    {
                        BinaryTreePath{{
                            BinaryTreePathEntry::LEFT_CHILD,
                        }},
                        mv_stride_1,
                    },
                    {
                        BinaryTreePath{{
                            BinaryTreePathEntry::RIGHT_CHILD,
                        }},
                        mv_stride_2,
                    },
                }},
            },
        };

        CHECK(result == correct);
      }

      SUBCASE("solution is running operators in parallel") {
        RuntimeOnlyCostEstimator runtime_only_cost_estimator =
            make_fake_runtime_only_cost_estimator(
                std::map<RuntimeOnlyOpCostEstimateKey,
                         RuntimeOnlyOpCostMetrics>{{
                    mk_cost_entry(k1, mv_stride_1, 1),
                    mk_cost_entry(k1, mv_stride_2, 3),
                    mk_cost_entry(k2, mv_stride_1, 3),
                    mk_cost_entry(k2, mv_stride_2, 4),
                }},
                std::map<TensorSetMovement, milliseconds_t>{{
                    {
                        TensorSetMovement{{}},
                        0.0_ms,
                    },
                }});

        MachineMappingContext context = MachineMappingContext{
            /*cost_estimator=*/runtime_only_cost_estimator,
            /*allowed_machine_views=*/allowed_machine_views,
        };

        MachineMappingResult result = get_optimal_machine_mapping(
            cache, context, problem_tree, four_nodes_resources, constraints);

        MachineView translated_mv_stride_1 = MachineView{
            /*start=*/MachineSpaceCoordinate{
                /*node_idx=*/2_n,
                /*device_idx=*/0_n,
            },
            /*dimensions=*/
            {
                MachineViewDimension{
                    /*stride=*/stride_t{1_p},
                    /*projection=*/MachineSpecificationDimension::INTER_NODE,
                },
            },
        };

        MachineMappingResult correct = MachineMappingResult{
            FeasibleMachineMappingResult{
                /*runtime=*/3_ms,
                /*machine_mapping=*/
                ParallelLayerGuidObliviousMachineMapping{{
                    {
                        BinaryTreePath{{
                            BinaryTreePathEntry::LEFT_CHILD,
                        }},
                        mv_stride_1,
                    },
                    {
                        BinaryTreePath{{
                            BinaryTreePathEntry::RIGHT_CHILD,
                        }},
                        translated_mv_stride_1,
                    },
                }},
            },
        };

        CHECK(result == correct);
      }

      SUBCASE("solution is running operators in series") {
        RuntimeOnlyCostEstimator runtime_only_cost_estimator =
            make_fake_runtime_only_cost_estimator(
                std::map<RuntimeOnlyOpCostEstimateKey,
                         RuntimeOnlyOpCostMetrics>{{
                    mk_cost_entry(k1, mv_stride_1, 3),
                    mk_cost_entry(k1, mv_stride_2, 1),
                    mk_cost_entry(k2, mv_stride_1, 4),
                    mk_cost_entry(k2, mv_stride_2, 1),
                }},
                std::map<TensorSetMovement, milliseconds_t>{{
                    {
                        TensorSetMovement{{}},
                        0.0_ms,
                    },
                }});

        MachineMappingContext context = MachineMappingContext{
            /*cost_estimator=*/runtime_only_cost_estimator,
            /*allowed_machine_views=*/allowed_machine_views,
        };

        MachineMappingResult result = get_optimal_machine_mapping(
            cache, context, problem_tree, four_nodes_resources, constraints);

        MachineMappingResult correct = MachineMappingResult{
            FeasibleMachineMappingResult{
                /*runtime=*/2_ms,
                /*machine_mapping=*/
                ParallelLayerGuidObliviousMachineMapping{{
                    {
                        BinaryTreePath{{
                            BinaryTreePathEntry::LEFT_CHILD,
                        }},
                        mv_stride_2,
                    },
                    {
                        BinaryTreePath{{
                            BinaryTreePathEntry::RIGHT_CHILD,
                        }},
                        mv_stride_2,
                    },
                }},
            },
        };

        CHECK(result == correct);
      }
    }
  }
}
