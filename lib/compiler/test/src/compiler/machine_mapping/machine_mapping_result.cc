#include "compiler/machine_mapping/machine_mapping_result.h"
#include "pcg/machine_view.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("series_combine") {
    MachineMemoryConstraints memory_constraints = MachineMemoryConstraints{
        /*memory_limit=*/10,
    };
    MachineMappingConfig config = MachineMappingConfig{
        /*enable_memory_optimization=*/false,
    };

    MachineView machine_view_0 = make_1d_machine_view(gpu_id_t(0), gpu_id_t(1));
    MachineView machine_view_1 = make_1d_machine_view(gpu_id_t(0), gpu_id_t(2));

    CostMetric pre_cost = CostMetric{
        /*runtime=*/2.0,
        /*memory=*/2,
    };
    MachineMappingResult pre = MachineMappingResult{
        FeasibleMachineMappingResult{
            /*runtime=*/pre_cost,
            /*machine_mapping=*/
            ParallelLayerGuidObliviousMachineMapping{{
                {
                    BinaryTreePath{{
                        BinaryTreePathEntry::LEFT_CHILD,
                    }},
                    machine_view_0,
                },
                {
                    BinaryTreePath{{
                        BinaryTreePathEntry::RIGHT_CHILD,
                    }},
                    machine_view_1,
                },
            }},
        },
    };

    CostMetric post_cost = CostMetric{
        /*runtime=*/4.0,
        /*memory=*/1,
    };
    MachineMappingResult post = MachineMappingResult{
        FeasibleMachineMappingResult{
            /*runtime=*/post_cost,
            /*machine_mapping=*/
            ParallelLayerGuidObliviousMachineMapping{{
                {
                    BinaryTreePath{{}},
                    machine_view_1,
                },
            }},
        },
    };

    MachineMappingResult infeasible = infeasible_machine_mapping_result();

    CostMetric comm_cost = CostMetric{
        /*runtime=*/3.0,
        /*memory=*/0,
    };

    SUBCASE("pre is infeasbile") {
      MachineMappingResult result =
          series_combine(config,
                         memory_constraints,
                         comm_cost,
                         infeasible,
                         post,
                         ParallelSplitTransformation::LthenR);
      MachineMappingResult correct = infeasible;

      CHECK(result == correct);
    }

    SUBCASE("post is infeasbile") {
      MachineMappingResult result =
          series_combine(config,
                         memory_constraints,
                         comm_cost,
                         pre,
                         infeasible,
                         ParallelSplitTransformation::LthenR);
      MachineMappingResult correct = infeasible;

      CHECK(result == correct);
    }

    SUBCASE("both are infeasible") {
      MachineMappingResult result =
          series_combine(config,
                         memory_constraints,
                         comm_cost,
                         infeasible,
                         infeasible,
                         ParallelSplitTransformation::LthenR);
      MachineMappingResult correct = infeasible;

      CHECK(result == correct);
    }

    SUBCASE("both are feasible") {
      CostMetric no_parallel_split_transform_cost = CostMetric{
          /*runtime=*/pre_cost.runtime + post_cost.runtime + comm_cost.runtime,
          /*memory=*/pre_cost.memory + post_cost.memory + comm_cost.memory,
      };
      MachineMappingResult no_parallel_split_transform = MachineMappingResult{
          FeasibleMachineMappingResult{
              /*cost=*/no_parallel_split_transform_cost,
              /*machine_mapping=*/
              ParallelLayerGuidObliviousMachineMapping{{
                  {
                      BinaryTreePath{{
                          BinaryTreePathEntry::LEFT_CHILD,
                          BinaryTreePathEntry::LEFT_CHILD,
                      }},
                      machine_view_0,
                  },
                  {
                      BinaryTreePath{{
                          BinaryTreePathEntry::LEFT_CHILD,
                          BinaryTreePathEntry::RIGHT_CHILD,
                      }},
                      machine_view_1,
                  },
                  {
                      BinaryTreePath{{
                          BinaryTreePathEntry::RIGHT_CHILD,
                      }},
                      machine_view_1,
                  },
              }},
          },
      };

      SUBCASE("parallel_split_transformation = std::nullopt") {
        MachineMappingResult result = series_combine(
            config, memory_constraints, comm_cost, pre, post, std::nullopt);
        MachineMappingResult correct = no_parallel_split_transform;

        CHECK(result == correct);
      }

      SUBCASE("parallel_split_transformation = LthenR") {
        MachineMappingResult result =
            series_combine(config,
                           memory_constraints,
                           comm_cost,
                           pre,
                           post,
                           ParallelSplitTransformation::LthenR);
        MachineMappingResult correct = no_parallel_split_transform;

        CHECK(result == correct);
      }

      SUBCASE("parallel_split_transformation = RthenL") {
        MachineMappingResult result =
            series_combine(config,
                           memory_constraints,
                           comm_cost,
                           pre,
                           post,
                           ParallelSplitTransformation::RthenL);
        CostMetric correct_cost = CostMetric{
            /*runtime=*/pre_cost.runtime + post_cost.runtime +
                comm_cost.runtime,
            /*memory=*/pre_cost.memory + post_cost.memory + comm_cost.memory,
        };
        MachineMappingResult correct = MachineMappingResult{
            FeasibleMachineMappingResult{
                /*runtime=*/correct_cost,
                /*machine_mapping=*/
                ParallelLayerGuidObliviousMachineMapping{{
                    {
                        BinaryTreePath{{
                            BinaryTreePathEntry::RIGHT_CHILD,
                            BinaryTreePathEntry::LEFT_CHILD,
                        }},
                        machine_view_0,
                    },
                    {
                        BinaryTreePath{{
                            BinaryTreePathEntry::RIGHT_CHILD,
                            BinaryTreePathEntry::RIGHT_CHILD,
                        }},
                        machine_view_1,
                    },
                    {
                        BinaryTreePath{{
                            BinaryTreePathEntry::LEFT_CHILD,
                        }},
                        machine_view_1,
                    },
                }},
            },
        };

        CHECK(result == correct);
      }
    }
  }

  TEST_CASE("parallel_combine") {
    MachineMemoryConstraints memory_constraints = MachineMemoryConstraints{
        /*memory_limit=*/10,
    };
    MachineMappingConfig config = MachineMappingConfig{
        /*enable_memory_optimization=*/false,
    };

    MachineView machine_view_0 = make_1d_machine_view(gpu_id_t(0), gpu_id_t(1));
    MachineView machine_view_1 = make_1d_machine_view(gpu_id_t(0), gpu_id_t(2));

    CostMetric lhs_cost = CostMetric{
        /*runtime=*/2.0,
        /*memory=*/2,
    };

    CostMetric rhs_cost = CostMetric{
        /*runtime=*/4.0,
        /*memory=*/1,
    };

    MachineMappingResult lhs = MachineMappingResult{
        FeasibleMachineMappingResult{
            /*cost=*/lhs_cost,
            /*machine_mapping=*/
            ParallelLayerGuidObliviousMachineMapping{{
                {
                    BinaryTreePath{{
                        BinaryTreePathEntry::LEFT_CHILD,
                    }},
                    machine_view_0,
                },
                {
                    BinaryTreePath{{
                        BinaryTreePathEntry::RIGHT_CHILD,
                    }},
                    machine_view_1,
                },
            }},
        },
    };

    MachineMappingResult rhs = MachineMappingResult{
        FeasibleMachineMappingResult{
            /*cost=*/rhs_cost,
            /*machine_mapping=*/
            ParallelLayerGuidObliviousMachineMapping{{
                {
                    BinaryTreePath{{}},
                    machine_view_1,
                },
            }},
        },
    };

    MachineMappingResult infeasible = infeasible_machine_mapping_result();

    SUBCASE("lhs is infeasbile") {
      MachineMappingResult result =
          parallel_combine(config, memory_constraints, infeasible, rhs);
      MachineMappingResult correct = infeasible;

      CHECK(result == correct);
    }

    SUBCASE("rhs is infeasbile") {
      MachineMappingResult result =
          parallel_combine(config, memory_constraints, lhs, infeasible);
      MachineMappingResult correct = infeasible;

      CHECK(result == correct);
    }

    SUBCASE("both are infeasible") {
      MachineMappingResult result =
          parallel_combine(config, memory_constraints, infeasible, infeasible);
      MachineMappingResult correct = infeasible;

      CHECK(result == correct);
    }

    SUBCASE("both are feasible") {
      MachineMappingResult result =
          parallel_combine(config, memory_constraints, lhs, rhs);

      CostMetric correct_cost = CostMetric{
          /*runtime=*/4.0,
          /*memory=*/2,
      };
      MachineMappingResult correct = MachineMappingResult{
          FeasibleMachineMappingResult{
              /*cost=*/correct_cost,
              /*machine_mapping=*/
              ParallelLayerGuidObliviousMachineMapping{{
                  {
                      BinaryTreePath{{
                          BinaryTreePathEntry::LEFT_CHILD,
                          BinaryTreePathEntry::LEFT_CHILD,
                      }},
                      machine_view_0,
                  },
                  {
                      BinaryTreePath{{
                          BinaryTreePathEntry::LEFT_CHILD,
                          BinaryTreePathEntry::RIGHT_CHILD,
                      }},
                      machine_view_1,
                  },
                  {
                      BinaryTreePath{{
                          BinaryTreePathEntry::RIGHT_CHILD,
                      }},
                      machine_view_1,
                  },
              }},
          },
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("minimize_runtime") {
    MachineView machine_view_0 = make_1d_machine_view(gpu_id_t(0), gpu_id_t(1));
    MachineView machine_view_1 = make_1d_machine_view(gpu_id_t(0), gpu_id_t(2));

    MachineMappingResult faster = MachineMappingResult{
        FeasibleMachineMappingResult{
            /*cost=*/CostMetric{2.0, 2},
            /*machine_mapping=*/
            ParallelLayerGuidObliviousMachineMapping{{
                {
                    BinaryTreePath{{
                        BinaryTreePathEntry::LEFT_CHILD,
                    }},
                    machine_view_0,
                },
                {
                    BinaryTreePath{{
                        BinaryTreePathEntry::RIGHT_CHILD,
                    }},
                    machine_view_1,
                },
            }},
        },
    };

    MachineMappingResult slower = MachineMappingResult{
        FeasibleMachineMappingResult{
            /*cost=*/CostMetric{4.0, 1},
            /*machine_mapping=*/
            ParallelLayerGuidObliviousMachineMapping{{
                {
                    BinaryTreePath{{}},
                    machine_view_1,
                },
            }},
        },
    };

    MachineMappingResult infeasible = infeasible_machine_mapping_result();

    SUBCASE("lhs is infeasbile") {
      MachineMappingResult result = minimize_runtime(infeasible, slower);
      MachineMappingResult correct = slower;

      CHECK(result == correct);
    }

    SUBCASE("rhs is infeasible") {
      MachineMappingResult result = minimize_runtime(slower, infeasible);
      MachineMappingResult correct = slower;

      CHECK(result == correct);
    }

    SUBCASE("both are infeasible") {
      MachineMappingResult result = minimize_runtime(infeasible, infeasible);
      MachineMappingResult correct = infeasible;

      CHECK(result == correct);
    }

    SUBCASE("both are feasible") {
      SUBCASE("lhs is faster") {
        MachineMappingResult result = minimize_runtime(faster, slower);
        MachineMappingResult correct = faster;

        CHECK(result == correct);
      }

      SUBCASE("rhs is faster") {
        MachineMappingResult result = minimize_runtime(slower, faster);
        MachineMappingResult correct = faster;

        CHECK(result == correct);
      }

      SUBCASE("lhs and rhs have the same speed") {
        MachineMappingResult result = minimize_runtime(slower, slower);
        MachineMappingResult correct = slower;

        CHECK(result == correct);
      }
    }
  }
}
