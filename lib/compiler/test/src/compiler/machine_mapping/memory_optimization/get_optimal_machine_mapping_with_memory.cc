#include "compiler/machine_mapping/memory_optimization/get_optimal_machine_mapping_with_memory.h"
#include "compiler/cost_estimator/tensor_set_movement.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_tensor_set_movement.h"
#include "compiler/machine_mapping/machine_mapping_constraints.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/machine_mapping_problem_tree.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/unmapped_op_cost_estimate_key.h"
#include "compiler/machine_mapping/machine_view.h"
#include "compiler/machine_mapping/memory_optimization/machine_mapping_with_memory_cache.h"
#include "internal/cost_estimator_for_test.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/task_space_coordinate.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "utils/containers/get_only.h"
#include "utils/containers/map_from_pairs.h"
#include "utils/full_binary_tree/binary_tree_path.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_optimal_machine_mapping_with_memory") {
    auto make_leaf = [](UnmappedOpCostEstimateKey const &k) {
      return MachineMappingProblemTree{
          runtime_only_from_unmapped_op_cost_estimate_key(k)};
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

    MachineView mv1 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/0_n,
            /*device_idx=*/0_n,
        },
        /*dimensions=*/{},
    };

    MachineView mv2 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/0_n,
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

    MachineView mv3 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/0_n,
            /*device_idx=*/0_n,
        },
        /*dimensions=*/
        {
            MachineViewDimension{
                /*stride=*/stride_t{2_p},
                /*projection=*/MachineSpecificationDimension::INTER_NODE,
            },
        },
    };

    MachineView mv4 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/1_n,
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
                12_p,
                8_p,
            },
        },
        DataType::FLOAT,
    };

    ParallelTensorShape pre_partition_par_tensor_shape =
        lift_to_parallel(tensor_shape);
    ParallelTensorShape post_partition_par_tensor_shape =
        lift_to_parallel_with_degrees(
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

    OptimizerAttrs optimizer_attrs = OptimizerAttrs{
        SGDOptimizerAttrs{
            /*lr=*/0.1,
            /*momentum=*/0.1,
            /*nesterov=*/false,
            /*weight_decay=*/0.1,
        },
    };

    UnmappedOpCostEstimateKey k1 = UnmappedOpCostEstimateKey{
        /*op_attrs=*/PCGOperatorAttrs{InputAttrs{tensor_shape}},
        /*input_shapes=*/{},
        /*weight_shapes=*/{},
        /*output_shapes=*/
        {
            {
                TensorSlotName::OUTPUT,
                pre_partition_par_tensor_shape,
            },
        },
        /*optimizer_attrs=*/optimizer_attrs,
    };

    UnmappedOpCostEstimateKey k2 = UnmappedOpCostEstimateKey{
        /*op_attrs=*/PCGOperatorAttrs{ElementUnaryAttrs{
            /*type=*/ElementUnaryOp::GELU,
            /*scalar=*/std::nullopt,
        }},
        /*input_shapes=*/
        {
            {
                TensorSlotName::INPUT,
                post_partition_par_tensor_shape,
            },
        },
        /*weight_shapes=*/{},
        /*output_shapes=*/
        {
            {
                TensorSlotName::OUTPUT,
                post_partition_par_tensor_shape,
            },
        },
        /*optimizer_attrs=*/optimizer_attrs,
    };

    UnmappedOpCostEstimateKey k3 = UnmappedOpCostEstimateKey{
        /*op_attrs=*/PCGOperatorAttrs{ElementUnaryAttrs{
            /*type=*/ElementUnaryOp::RELU,
            /*scalar=*/std::nullopt,
        }},
        /*input_shapes=*/
        {
            {
                TensorSlotName::INPUT,
                post_partition_par_tensor_shape,
            },
        },
        /*weight_shapes=*/{},
        /*output_shapes=*/
        {
            {
                TensorSlotName::OUTPUT,
                post_partition_par_tensor_shape,
            },
        },
        /*optimizer_attrs=*/optimizer_attrs,
    };

    TaskSpaceCoordinate empty_task_space_coord =
        TaskSpaceCoordinate{OrthotopeCoord{{}}};

    BinaryTreePath src_path = binary_tree_root_path();
    TaskSpaceCoordinate src_coord = empty_task_space_coord;

    AbstractedDevice dst_device = AbstractedDevice{
        /*operator_tree_path=*/binary_tree_root_path(),
        /*task_space_coordinate=*/empty_task_space_coord,
    };

    ParallelLayerGuidObliviousMachineMapping mm1 =
        ParallelLayerGuidObliviousMachineMapping{{
            {binary_tree_root_path(), mv1},
        }};
    ParallelLayerGuidObliviousMachineMapping mm2 =
        ParallelLayerGuidObliviousMachineMapping{{
            {binary_tree_root_path(), mv2},
        }};

    OperatorTaskSpace unparallel_task_space =
        OperatorTaskSpace{MinimalOrthotope{{}}};
    OperatorTaskSpace parallel_task_space = OperatorTaskSpace{
        MinimalOrthotope{{
            2_ge2,
        }},
    };

    auto get_corresponding_task_space = [&](MachineView const &mv) {
      if (mv == mv1) {
        return unparallel_task_space;
      } else {
        ASSERT(mv == mv2 || mv == mv3);

        return parallel_task_space;
      }
    };

    SUBCASE("single layer with single option") {
      OpCostMetrics k1_on_mv1_cost = OpCostMetrics{
          /*forward_runtime=*/1_ms,
          /*backward_runtime=*/1_ms,
          /*memory_usage=*/2_bytes,
      };

      CostEstimator cost_estimator = make_fake_cost_estimator(
          std::map<OpCostEstimateKey, OpCostMetrics>{{
              {
                  map_unmapped_op_cost_estimate_key(k1, mv1),
                  k1_on_mv1_cost,
              },
          }},
          std::map<TensorSetMovement, milliseconds_t>{
              {
                  empty_tensor_set_movement(),
                  0_ms,
              },
          });

      MachineMappingProblemTree problem_tree = make_leaf(k1);

      MachineMappingConstraints constraints =
          get_unconstrained_solution_for_layers(
              get_all_leaf_paths(problem_tree));

      auto allowed_machine_views =
          [&](UnmappedRuntimeOnlyOpCostEstimateKey const &k,
              MachineComputeResourceSlice const &resources) {
            ASSERT(k == runtime_only_from_unmapped_op_cost_estimate_key(k1));
            ASSERT(resources == four_nodes_resources);
            return std::set<MachineView>{mv1};
          };

      MachineMappingWithMemoryContext context = MachineMappingWithMemoryContext{
          cost_estimator,
          optimizer_attrs,
          allowed_machine_views,
      };

      MachineMappingWithMemoryCache cache =
          empty_machine_mapping_with_memory_cache();

      MachineMappingWithMemoryResult result =
          get_optimal_machine_mapping_with_memory(
              cache, context, problem_tree, four_nodes_resources, constraints);

      MachineMappingWithMemoryResult correct = MachineMappingWithMemoryResult{{
          ParetoOptimalMachineMapping{
              k1_on_mv1_cost,
              ParallelLayerGuidObliviousMachineMapping{{
                  {binary_tree_root_path(), mv1},
              }},
          },
      }};

      CHECK(result == correct);
    }

    SUBCASE("single layer with multiple options") {

      auto allowed_machine_views =
          [&](UnmappedRuntimeOnlyOpCostEstimateKey const &k,
              MachineComputeResourceSlice const &resources) {
            ASSERT(k == runtime_only_from_unmapped_op_cost_estimate_key(k3));
            ASSERT(resources == four_nodes_resources);
            return std::set<MachineView>{mv2, mv3, mv4};
          };

      OpCostMetrics k3_on_mv2_cost = OpCostMetrics{
          /*forward_runtime=*/2.5_ms,
          /*backward_runtime=*/2.5_ms,
          /*memory_usage=*/2_bytes,
      };

      OpCostMetrics k3_on_mv3_cost = OpCostMetrics{
          /*forward_runtime=*/2_ms,
          /*backward_runtime=*/2_ms,
          /*memory_usage=*/2_bytes,
      };

      OpCostMetrics k3_on_mv4_cost = OpCostMetrics{
          /*forward_runtime=*/3_ms,
          /*backward_runtime=*/3_ms,
          /*memory_usage=*/3_bytes,
      };

      CostEstimator cost_estimator = make_fake_cost_estimator(
          std::map<OpCostEstimateKey, OpCostMetrics>{{
              {
                  map_unmapped_op_cost_estimate_key(k3, mv2),
                  k3_on_mv2_cost,
              },
              {
                  map_unmapped_op_cost_estimate_key(k3, mv3),
                  k3_on_mv3_cost,
              },
              {
                  map_unmapped_op_cost_estimate_key(k3, mv4),
                  k3_on_mv4_cost,
              },
          }},
          std::map<TensorSetMovement, milliseconds_t>{
              {
                  empty_tensor_set_movement(),
                  0_ms,
              },
          });

      MachineMappingProblemTree problem_tree = make_leaf(k3);

      MachineMappingConstraints constraints =
          get_unconstrained_solution_for_layers(
              get_all_leaf_paths(problem_tree));

      MachineMappingWithMemoryCache cache =
          empty_machine_mapping_with_memory_cache();

      MachineMappingWithMemoryContext context = MachineMappingWithMemoryContext{
          cost_estimator,
          optimizer_attrs,
          allowed_machine_views,
      };

      MachineMappingWithMemoryResult result =
          get_optimal_machine_mapping_with_memory(
              cache, context, problem_tree, four_nodes_resources, constraints);

      MachineMappingWithMemoryResult correct = MachineMappingWithMemoryResult{{
          ParetoOptimalMachineMapping{
              k3_on_mv2_cost,
              ParallelLayerGuidObliviousMachineMapping{{
                  {binary_tree_root_path(), mv2},
              }},
          },
          ParetoOptimalMachineMapping{
              k3_on_mv3_cost,
              ParallelLayerGuidObliviousMachineMapping{{
                  {binary_tree_root_path(), mv3},
              }},
          },
      }};

      CHECK(result == correct);
    }

    SUBCASE("pair of layers in sequence") {
      AbstractedTensorSetMovement k2_to_k3 = AbstractedTensorSetMovement{
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

      auto mk_tensor_set_movement = [&](MachineView const &src_mv,
                                        MachineView const &dst_mv) {
        MachineSpaceStencil src_stencil = MachineSpaceStencil{
            /*operator_task_space=*/get_corresponding_task_space(src_mv),
            /*machine_view=*/src_mv,
        };

        MachineSpaceStencil dst_stencil = MachineSpaceStencil{
            /*operator_task_space=*/get_corresponding_task_space(dst_mv),
            /*machine_view=*/dst_mv,
        };

        return concretize_abstracted_tensor_set_movement(
            k2_to_k3,
            /*pre_machine_stencils=*/{{binary_tree_root_path(), src_stencil}},
            /*post_machine_stencils=*/{{binary_tree_root_path(), dst_stencil}});
      };

      auto mk_cost_estimator = [&](milliseconds_t k2_on_mv2_cost,
                                   num_bytes_t k2_on_mv2_mem_usage,
                                   milliseconds_t k2_on_mv3_cost,
                                   num_bytes_t k2_on_mv3_mem_usage,
                                   milliseconds_t k3_on_mv2_cost,
                                   num_bytes_t k3_on_mv2_mem_usage,
                                   milliseconds_t k3_on_mv3_cost,
                                   num_bytes_t k3_on_mv3_mem_usage,
                                   milliseconds_t mv2_to_mv2_cost,
                                   milliseconds_t mv2_to_mv3_cost,
                                   milliseconds_t mv3_to_mv2_cost,
                                   milliseconds_t mv3_to_mv3_cost) {
        return make_fake_cost_estimator(
            std::map<OpCostEstimateKey, OpCostMetrics>{{
                {
                    map_unmapped_op_cost_estimate_key(k2, mv2),
                    OpCostMetrics{
                        /*forward_runtime=*/k2_on_mv2_cost,
                        /*backward_runtime=*/k2_on_mv2_cost,
                        /*memory_usage=*/k2_on_mv2_mem_usage,
                    },
                },
                {
                    map_unmapped_op_cost_estimate_key(k2, mv3),
                    OpCostMetrics{
                        /*forward_runtime=*/k2_on_mv3_cost,
                        /*backward_runtime=*/k2_on_mv3_cost,
                        /*memory_usage=*/k2_on_mv3_mem_usage,
                    },
                },
                {
                    map_unmapped_op_cost_estimate_key(k3, mv2),
                    OpCostMetrics{
                        /*forward_runtime=*/k3_on_mv2_cost,
                        /*backward_runtime=*/k3_on_mv2_cost,
                        /*memory_usage=*/k3_on_mv2_mem_usage,
                    },
                },
                {
                    map_unmapped_op_cost_estimate_key(k3, mv3),
                    OpCostMetrics{
                        /*forward_runtime=*/k3_on_mv3_cost,
                        /*backward_runtime=*/k3_on_mv3_cost,
                        /*memory_usage=*/k3_on_mv3_mem_usage,
                    },
                },
            }},
            std::map<TensorSetMovement, milliseconds_t>{{
                {
                    empty_tensor_set_movement(),
                    0_ms,
                },
                {
                    mk_tensor_set_movement(mv2, mv2),
                    mv2_to_mv2_cost,
                },
                {
                    mk_tensor_set_movement(mv2, mv3),
                    mv2_to_mv3_cost,
                },
                {
                    mk_tensor_set_movement(mv3, mv2),
                    mv3_to_mv2_cost,
                },
                {
                    mk_tensor_set_movement(mv3, mv3),
                    mv3_to_mv3_cost,
                },
            }});
      };

      MachineMappingProblemTree problem_tree =
          make_series_split(k2_to_k3, make_leaf(k2), make_leaf(k3));

      MachineMappingConstraints constraints =
          get_unconstrained_solution_for_layers(
              get_all_leaf_paths(problem_tree));

      MachineMappingWithMemoryCache cache =
          empty_machine_mapping_with_memory_cache();

      SUBCASE("solution is mv2, mv3 due to runtime") {
        CostEstimator cost_estimator = mk_cost_estimator(
            /*k2_on_mv2_cost=*/2_ms,
            /*k2_on_mv2_mem_usage=*/2_bytes,
            /*k2_on_mv3_cost=*/2.4_ms,
            /*k2_on_mv3_mem_usage=*/2_bytes,
            /*k3_on_mv2_cost=*/3.6_ms,
            /*k3_on_mv2_mem_usage=*/2_bytes,
            /*k3_on_mv3_cost=*/3_ms,
            /*k3_on_mv3_mem_usage=*/2_bytes,
            /*mv2_to_mv2_cost=*/0.1_ms,
            /*mv2_to_mv3_cost=*/1.0_ms,
            /*mv3_to_mv2_cost=*/0.3_ms,
            /*mv3_to_mv3_cost=*/0.1_ms);

        auto allowed_machine_views =
            [&](UnmappedRuntimeOnlyOpCostEstimateKey const &k,
                MachineComputeResourceSlice const &resources) {
              if (k == runtime_only_from_unmapped_op_cost_estimate_key(k1)) {
                return std::set<MachineView>{
                    mv1,
                };
              } else {
                if (resources == four_nodes_resources) {
                  return std::set<MachineView>{mv2, mv3};
                } else if (resources == three_nodes_resources) {
                  return std::set<MachineView>{mv2, mv3};
                } else if (resources == two_nodes_resources) {
                  return std::set<MachineView>{mv2};
                } else {
                  return std::set<MachineView>{};
                }
              };
            };

        MachineMappingWithMemoryContext context =
            MachineMappingWithMemoryContext{
                cost_estimator,
                optimizer_attrs,
                allowed_machine_views,
            };

        MachineMappingWithMemoryResult result =
            get_optimal_machine_mapping_with_memory(cache,
                                                    context,
                                                    problem_tree,
                                                    four_nodes_resources,
                                                    constraints);

        MachineMappingWithMemoryResult correct =
            MachineMappingWithMemoryResult{{
                ParetoOptimalMachineMapping{
                    OpCostMetrics{
                        /*forward_runtime=*/2_ms + 0.3_ms + 3_ms,
                        /*backward_runtime=*/2_ms + 0.3_ms + 3_ms,
                        /*memory_usage=*/4_bytes,
                    },
                    ParallelLayerGuidObliviousMachineMapping{{
                        {
                            BinaryTreePath{{
                                BinaryTreePathEntry::LEFT_CHILD,
                            }},
                            mv1,
                        },
                        {
                            BinaryTreePath{{
                                BinaryTreePathEntry::RIGHT_CHILD,
                            }},
                            mv1,
                        },
                    }},
                },
            }};
      }
    }

    SUBCASE("pair of layers in parallel") {
      auto mk_cost_estimator = [&](milliseconds_t k2_on_mv2_cost,
                                   num_bytes_t k2_on_mv2_mem_usage,
                                   milliseconds_t k2_on_mv3_cost,
                                   num_bytes_t k2_on_mv3_mem_usage,
                                   milliseconds_t k3_on_mv2_cost,
                                   num_bytes_t k3_on_mv2_mem_usage,
                                   milliseconds_t k3_on_mv3_cost,
                                   num_bytes_t k3_on_mv3_mem_usage) {
        return make_fake_cost_estimator(
            std::map<OpCostEstimateKey, OpCostMetrics>{{
                {
                    map_unmapped_op_cost_estimate_key(k2, mv2),
                    OpCostMetrics{
                        /*forward_runtime=*/k2_on_mv2_cost,
                        /*backward_runtime=*/k2_on_mv2_cost,
                        /*memory_usage=*/k2_on_mv2_mem_usage,
                    },
                },
                {
                    map_unmapped_op_cost_estimate_key(k2, mv3),
                    OpCostMetrics{
                        /*forward_runtime=*/k2_on_mv3_cost,
                        /*backward_runtime=*/k2_on_mv3_cost,
                        /*memory_usage=*/k2_on_mv3_mem_usage,
                    },
                },
                {
                    map_unmapped_op_cost_estimate_key(k3, mv2),
                    OpCostMetrics{
                        /*forward_runtime=*/k3_on_mv2_cost,
                        /*backward_runtime=*/k3_on_mv2_cost,
                        /*memory_usage=*/k3_on_mv2_mem_usage,
                    },
                },
                {
                    map_unmapped_op_cost_estimate_key(k3, mv3),
                    OpCostMetrics{
                        /*forward_runtime=*/k3_on_mv3_cost,
                        /*backward_runtime=*/k3_on_mv3_cost,
                        /*memory_usage=*/k3_on_mv3_mem_usage,
                    },
                },
            }},
            std::map<TensorSetMovement, milliseconds_t>{
                {
                    empty_tensor_set_movement(),
                    0_ms,
                },
            });
      };

      CostEstimator cost_estimator = mk_cost_estimator(
          /*k2_on_mv2_cost=*/2_ms,
          /*k2_on_mv2_mem_usage=*/3_bytes,
          /*k2_on_mv3_cost=*/2.5_ms,
          /*k2_on_mv3_mem_usage=*/2_bytes,
          /*k3_on_mv2_cost=*/2.5_ms,
          /*k3_on_mv2_mem_usage=*/2_bytes,
          /*k3_on_mv3_cost=*/2_ms,
          /*k3_on_mv3_mem_usage=*/1_bytes);

      auto allowed_machine_views =
          [&](UnmappedRuntimeOnlyOpCostEstimateKey const &k,
              MachineComputeResourceSlice const &resources) {
            if (k == runtime_only_from_unmapped_op_cost_estimate_key(k1)) {
              return std::set<MachineView>{
                  mv1,
              };
            } else {
              if (resources == four_nodes_resources) {
                return std::set<MachineView>{mv2, mv3};
              } else if (resources == three_nodes_resources) {
                return std::set<MachineView>{mv2, mv3};
              } else if (resources == two_nodes_resources) {
                return std::set<MachineView>{mv2};
              } else {
                return std::set<MachineView>{};
              }
            };
          };

      MachineMappingWithMemoryContext context = MachineMappingWithMemoryContext{
          cost_estimator,
          optimizer_attrs,
          allowed_machine_views,
      };

      MachineMappingProblemTree problem_tree =
          make_parallel_split(make_leaf(k2), make_leaf(k3));

      MachineMappingConstraints constraints =
          get_unconstrained_solution_for_layers(
              get_all_leaf_paths(problem_tree));

      MachineMappingWithMemoryCache cache =
          empty_machine_mapping_with_memory_cache();

      MachineMappingWithMemoryResult result =
          get_optimal_machine_mapping_with_memory(
              cache, context, problem_tree, four_nodes_resources, constraints);

      MachineView translated_mv2 = MachineView{
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

      MachineMappingWithMemoryResult correct = MachineMappingWithMemoryResult{
          /*pareto_frontier=*/{
              ParetoOptimalMachineMapping{
                  OpCostMetrics{
                      /*forward_runtime=*/2.5_ms,
                      /*backward_runtime=*/2.5_ms,
                      /*memory_usage=*/3_bytes,
                  },
                  ParallelLayerGuidObliviousMachineMapping{{
                      {
                          BinaryTreePath{{
                              BinaryTreePathEntry::LEFT_CHILD,
                          }},
                          mv2,
                      },
                      {
                          BinaryTreePath{{
                              BinaryTreePathEntry::RIGHT_CHILD,
                          }},
                          translated_mv2,
                      },
                  }},
              },
              ParetoOptimalMachineMapping{
                  OpCostMetrics{
                      /*forward_runtime=*/4.5_ms,
                      /*backward_runtime=*/4.5_ms,
                      /*memory_usage=*/3_bytes,
                  },
                  ParallelLayerGuidObliviousMachineMapping{{
                      {
                          BinaryTreePath{{
                              BinaryTreePathEntry::LEFT_CHILD,
                          }},
                          mv3,
                      },
                      {
                          BinaryTreePath{{
                              BinaryTreePathEntry::RIGHT_CHILD,
                          }},
                          mv3,
                      },
                  }},
              },
          },
      };

      ASSERT(result == correct);
    }
  }
}
