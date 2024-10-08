#include "compiler/machine_mapping/machine_mapping_result.h"
#include "pcg/machine_view.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("series_combine") {
    MachineView machine_view_0 = make_1d_machine_view(gpu_id_t(0), gpu_id_t(1));
    MachineView machine_view_1 = make_1d_machine_view(gpu_id_t(0), gpu_id_t(2));

    float pre_cost = 2.0;
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

    float post_cost = 4.0;
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

    float comm_cost = 3.0;

    SUBCASE("pre is infeasbile") {
      MachineMappingResult result = series_combine(
          comm_cost, infeasible, post, ParallelSplitTransformation::LthenR);
      MachineMappingResult correct = infeasible;

      CHECK(result == correct);
    }

    SUBCASE("post is infeasbile") {
      MachineMappingResult result = series_combine(
          comm_cost, pre, infeasible, ParallelSplitTransformation::LthenR);
      MachineMappingResult correct = infeasible;

      CHECK(result == correct);
    }

    SUBCASE("both are infeasible") {
      MachineMappingResult result =
          series_combine(comm_cost,
                         infeasible,
                         infeasible,
                         ParallelSplitTransformation::LthenR);
      MachineMappingResult correct = infeasible;

      CHECK(result == correct);
    }

    SUBCASE("both are feasible") {
      MachineMappingResult no_parallel_split_transform = MachineMappingResult{
          FeasibleMachineMappingResult{
              /*runtime=*/pre_cost + comm_cost + post_cost,
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
        MachineMappingResult result =
            series_combine(comm_cost, pre, post, std::nullopt);
        MachineMappingResult correct = no_parallel_split_transform;

        CHECK(result == correct);
      }

      SUBCASE("parallel_split_transformation = LthenR") {
        MachineMappingResult result = series_combine(
            comm_cost, pre, post, ParallelSplitTransformation::LthenR);
        MachineMappingResult correct = no_parallel_split_transform;

        CHECK(result == correct);
      }

      SUBCASE("parallel_split_transformation = RthenL") {
        MachineMappingResult result = series_combine(
            comm_cost, pre, post, ParallelSplitTransformation::RthenL);
        MachineMappingResult correct = MachineMappingResult{
            FeasibleMachineMappingResult{
                /*runtime=*/pre_cost + comm_cost + post_cost,
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
    MachineView machine_view_0 = make_1d_machine_view(gpu_id_t(0), gpu_id_t(1));
    MachineView machine_view_1 = make_1d_machine_view(gpu_id_t(0), gpu_id_t(2));

    MachineMappingResult lhs = MachineMappingResult{
        FeasibleMachineMappingResult{
            /*runtime=*/2.0,
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
            /*runtime=*/4.0,
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
      MachineMappingResult result = parallel_combine(infeasible, rhs);
      MachineMappingResult correct = infeasible;

      CHECK(result == correct);
    }

    SUBCASE("rhs is infeasbile") {
      MachineMappingResult result = parallel_combine(lhs, infeasible);
      MachineMappingResult correct = infeasible;

      CHECK(result == correct);
    }

    SUBCASE("both are infeasible") {
      MachineMappingResult result = parallel_combine(infeasible, infeasible);
      MachineMappingResult correct = infeasible;

      CHECK(result == correct);
    }

    SUBCASE("both are feasible") {
      MachineMappingResult result = parallel_combine(lhs, rhs);
      MachineMappingResult correct = MachineMappingResult{
          FeasibleMachineMappingResult{
              /*runtime=*/4.0,
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
            /*runtime=*/2.0,
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
            /*runtime=*/4.0,
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
