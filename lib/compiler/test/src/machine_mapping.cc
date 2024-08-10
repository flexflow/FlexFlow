#include "compiler/machine_mapping.h"
#include "doctest/doctest.h"
#include "pcg/machine_specification.dtg.h"
#include "test_generator.h"
#include "utils/containers/extend.h"
#include "utils/containers/is_subseteq_of.h"
#include "utils/containers/set_difference.h"

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("get_allowed_machine_view") {
    SUBCASE("no parallelism") {}

    SUBCASE("1 degree of parallelism") {
      MachineSpecification ms = MachineSpecification(5, 1, 1, 0, 0);
      ParallelTensorShape shape = ParallelTensorShape{
          ParallelTensorDims{
              FFOrdered<ShardParallelDim>{
                  ShardParallelDim{10, 3},
              },
              ReplicaParallelDimSet{
                  SumDegree{1},
                  DiscardCopyDegree{1},
              },
          },
          DataType::FLOAT,
      };

      std::unordered_set<MachineView> correct = {
          make_1d_machine_view(gpu_id_t(0), gpu_id_t(3), stride_t(1)),
          make_1d_machine_view(gpu_id_t(1), gpu_id_t(4), stride_t(1)),
          make_1d_machine_view(gpu_id_t(2), gpu_id_t(5), stride_t(1)),
          make_1d_machine_view(gpu_id_t(0), gpu_id_t(6), stride_t(2))};
      std::unordered_set<MachineView> result =
          get_allowed_machine_views(ms, shape);
      CHECK(correct == result);
    }

    SUBCASE("2 degrees of parallelism") {
      MachineSpecification ms = MachineSpecification(18, 1, 1, 0, 0);
      ParallelTensorShape shape = ParallelTensorShape{
          ParallelTensorDims{
              FFOrdered<ShardParallelDim>{
                  ShardParallelDim{10, 3},
              },
              ReplicaParallelDimSet{
                  SumDegree{2},
                  DiscardCopyDegree{1},
              },
          },
          DataType::FLOAT,
      };

      auto make_2d_views = [&](int num_starts, int stride1, int stride2) {
        std::unordered_set<MachineView> views;
        for (int i = 0; i < num_starts; i++) {
          StridedRectangle rect = StridedRectangle{
              {StridedRectangleSide{num_points_t(2), stride_t(stride1)},
               StridedRectangleSide{num_points_t(3), stride_t(stride2)}}};
          MachineView mv = MachineView{device_id_t(gpu_id_t(i)), rect};
          views.insert(mv);
        }
        return views;
      };
      std::unordered_set<MachineView> correct;
      extend(correct,
             make_2d_views(/*num_starts*/ 13, /*stride1*/ 1, /*stride2*/ 1));
      extend(correct,
             make_2d_views(/*num_starts*/ 8, /*stride1*/ 2, /*stride2*/ 1));
      extend(correct,
             make_2d_views(/*num_starts*/ 9, /*stride1*/ 1, /*stride2*/ 2));
      extend(correct,
             make_2d_views(/*num_starts*/ 3, /*stride1*/ 3, /*stride2*/ 1));
      extend(correct,
             make_2d_views(/*num_starts*/ 5, /*stride1*/ 1, /*stride2*/ 3));
      extend(correct,
             make_2d_views(/*num_starts*/ 1, /*stride1*/ 1, /*stride2*/ 4));

      std::unordered_set<MachineView> result =
          get_allowed_machine_views(ms, shape);
      CHECK(result == correct);
    }
  }

  // TEST_CASE("MachineMapping::combine") {
  //   RC_SUBCASE([](MachineMapping const &m0, MachineMapping const &m1) {
  //     RC_PRE(MachineMapping::nodes_are_disjoint(m0, m1));

  //     MachineMapping comb = MachineMapping::combine(m0, m1);

  //     RC_ASSERT(comb.machine_views.size() ==
  //               m0.machine_views.size() + m1.machine_views.size());
  //     RC_ASSERT(is_submap(comb.machine_views, m0.machine_views));
  //     RC_ASSERT(is_submap(comb.machine_views, m1.machine_views));
  //   });
  // }

  // TEST_CASE("OptimalCostResult::infinity") {
  //   RC_SUBCASE([](OptimalCostResult const &c) {
  //     RC_ASSERT(c.runtime <= OptimalCostResult::infinity().runtime);
  //   });
  // }
}
