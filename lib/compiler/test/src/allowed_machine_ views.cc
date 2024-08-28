#include "compiler/allowed_machine_views.h"
#include "doctest/doctest.h"
#include "pcg/machine_specification.dtg.h"
#include "pcg/machine_view.h"
#include "pcg/start_invariant_machine_view.h"
#include "utils/containers/extend.h"
#include "utils/containers/range.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_set_of.h"

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("get_allowed_machine_views") {

    SUBCASE("1 degree of parallelism") {

      MachineSpecification ms = MachineSpecification{5, 1, 1, 0, 0};
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

      MachineSpecification ms = MachineSpecification{11, 1, 1, 0, 0};
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
        return unordered_set_of(transform(range(num_starts), [&](int start) {
          return MachineView{
              device_id_t{gpu_id_t{start}},
              StridedRectangle{
                  {StridedRectangleSide{num_points_t{2}, stride_t{stride1}},
                   StridedRectangleSide{num_points_t{3}, stride_t{stride2}}}},
          };
        }));
      };

      std::unordered_set<MachineView> correct;
      extend(correct,
             make_2d_views(/*num_starts*/ 6, /*stride1*/ 1, /*stride2*/ 1));
      extend(correct,
             make_2d_views(/*num_starts*/ 1, /*stride1*/ 2, /*stride2*/ 1));
      extend(correct,
             make_2d_views(/*num_starts*/ 2, /*stride1*/ 1, /*stride2*/ 2));

      std::unordered_set<MachineView> result =
          get_allowed_machine_views(ms, shape);

      CHECK(result == correct);
    }
  }

  TEST_CASE("get_allowed_start_invariant_machine_views") {

    SUBCASE("1 degree of parallelism") {

      MachineSpecification ms = MachineSpecification{5, 1, 1, 0, 0};
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

      std::unordered_set<StartInvariantMachineView> correct = {
          make_1d_start_invariant_machine_view(num_points_t(3), stride_t(1)),
          make_1d_start_invariant_machine_view(num_points_t(3), stride_t(2))};
      std::unordered_set<StartInvariantMachineView> result =
          get_allowed_start_invariant_machine_views(ms, shape);

      CHECK(correct == result);
    }

    SUBCASE("2 degrees of parallelism") {

      MachineSpecification ms = MachineSpecification(15, 1, 1, 0, 0);
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

      auto make_2d_view = [&](int stride1, int stride2) {
        StridedRectangle rect = StridedRectangle{
            {StridedRectangleSide{num_points_t(2), stride_t(stride1)},
             StridedRectangleSide{num_points_t(3), stride_t(stride2)}}};
        return StartInvariantMachineView{rect};
      };

      std::unordered_set<StartInvariantMachineView> correct = {
          make_2d_view(/*stride1*/ 1, /*stride2*/ 1),
          make_2d_view(/*stride1*/ 2, /*stride2*/ 1),
          make_2d_view(/*stride1*/ 1, /*stride2*/ 2),
          make_2d_view(/*stride1*/ 1, /*stride2*/ 3),
      };

      std::unordered_set<StartInvariantMachineView> result =
          get_allowed_start_invariant_machine_views(ms, shape);

      CHECK(result == correct);
    }
  }
}
