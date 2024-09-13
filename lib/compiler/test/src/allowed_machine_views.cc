#include "compiler/allowed_machine_views.h"
#include "doctest/doctest.h"
#include "pcg/machine_specification.dtg.h"
#include "pcg/machine_view.h"
#include "pcg/start_invariant_machine_view.h"
#include "utils/containers/extend.h"
#include "utils/containers/range.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/containers/zip.h"
#include "utils/fmt/unordered_set.h"

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("get_allowed_partial_machine_view_mappings") {

    SUBCASE("1 degree of parallelism") {

      MachineSpecification ms = MachineSpecification{1, 5, 5, 0, 0};
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

      std::vector<MachineView> correct_mv = {
          make_1d_machine_view(0, 3, stride_t(1)),
          make_1d_machine_view(1, 4, stride_t(1)),
          make_1d_machine_view(2, 5, stride_t(1)),
          make_1d_machine_view(0, 6, stride_t(2))};

      std::vector<MachineViewProjection> correct_proj = {
          MachineViewProjection{{{machine_view_dim_idx_t{0},
                                  MachineSpecificationDimension::INTRA}}},
          MachineViewProjection{{{machine_view_dim_idx_t{0},
                                  MachineSpecificationDimension::INTRA}}},
          MachineViewProjection{{{machine_view_dim_idx_t{0},
                                  MachineSpecificationDimension::INTRA}}},
          MachineViewProjection{{{machine_view_dim_idx_t{0},
                                  MachineSpecificationDimension::INTRA}}},
      };

      std::unordered_set<std::pair<MachineView, MachineViewProjection>>
          correct = unordered_set_of(zip(correct_mv, correct_proj));

      std::unordered_set<std::pair<MachineView, MachineViewProjection>> result =
          get_allowed_partial_machine_view_mappings(ms, shape);

      CHECK(correct == result);
    }

    SUBCASE("2 degrees of parallelism") {

      MachineSpecification ms = MachineSpecification{3, 3, 3, 0, 0};
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

      auto make_2d_views =
          [&](int start_x, int start_y, int stride1, int stride2) {
            return MachineView{
                MachineViewCoordinates{{start_x, start_y}},
                StridedRectangle{
                    {StridedRectangleSide{num_points_t{2}, stride_t{stride1}},
                     StridedRectangleSide{num_points_t{3}, stride_t{stride2}}}},
                DeviceType::GPU};
          };

      std::vector<MachineView> correct_mv = {
          make_2d_views(0, 0, /*stride1*/ 1, /*stride2*/ 1),
          make_2d_views(1, 0, /*stride1*/ 1, /*stride2*/ 1),
          make_2d_views(0, 0, /*stride1*/ 2, /*stride2*/ 1),

          make_2d_views(0, 0, /*stride1*/ 1, /*stride2*/ 1),
          make_2d_views(1, 0, /*stride1*/ 1, /*stride2*/ 1),
          make_2d_views(0, 0, /*stride1*/ 2, /*stride2*/ 1),
      };

      std::vector<MachineViewProjection> correct_proj = {
          MachineViewProjection{{{machine_view_dim_idx_t{0},
                                  MachineSpecificationDimension::INTRA},
                                 {machine_view_dim_idx_t{1},
                                  MachineSpecificationDimension::INTER}}},
          MachineViewProjection{{{machine_view_dim_idx_t{0},
                                  MachineSpecificationDimension::INTRA},
                                 {machine_view_dim_idx_t{1},
                                  MachineSpecificationDimension::INTER}}},
          MachineViewProjection{{{machine_view_dim_idx_t{0},
                                  MachineSpecificationDimension::INTRA},
                                 {machine_view_dim_idx_t{1},
                                  MachineSpecificationDimension::INTER}}},

          MachineViewProjection{{{machine_view_dim_idx_t{0},
                                  MachineSpecificationDimension::INTER},
                                 {machine_view_dim_idx_t{1},
                                  MachineSpecificationDimension::INTRA}}},
          MachineViewProjection{{{machine_view_dim_idx_t{0},
                                  MachineSpecificationDimension::INTER},
                                 {machine_view_dim_idx_t{1},
                                  MachineSpecificationDimension::INTRA}}},
          MachineViewProjection{{{machine_view_dim_idx_t{0},
                                  MachineSpecificationDimension::INTER},
                                 {machine_view_dim_idx_t{1},
                                  MachineSpecificationDimension::INTRA}}},
      };

      std::unordered_set<std::pair<MachineView, MachineViewProjection>>
          correct = unordered_set_of(zip(correct_mv, correct_proj));

      std::unordered_set<std::pair<MachineView, MachineViewProjection>> result =
          get_allowed_partial_machine_view_mappings(ms, shape, DeviceType::GPU);

      CHECK(correct == result);
    }
  }

  TEST_CASE("get_allowed_partial_start_invariant_machine_view_mappings") {

    SUBCASE("1 degree of parallelism") {

      MachineSpecification ms = MachineSpecification{1, 5, 5, 0, 0};
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

      std::vector<StartInvariantMachineView> correct_mv = {
          make_1d_start_invariant_machine_view(
              num_points_t(3), stride_t(1), DeviceType::GPU),
          make_1d_start_invariant_machine_view(
              num_points_t(3), stride_t(2), DeviceType::GPU)};

      std::vector<MachineViewProjection> correct_proj = {
          MachineViewProjection{{{machine_view_dim_idx_t{0},
                                  MachineSpecificationDimension::INTRA}}},
          MachineViewProjection{{{machine_view_dim_idx_t{0},
                                  MachineSpecificationDimension::INTRA}}},
      };

      std::unordered_set<
          std::pair<StartInvariantMachineView, MachineViewProjection>>
          correct = unordered_set_of(zip(correct_mv, correct_proj));

      std::unordered_set<
          std::pair<StartInvariantMachineView, MachineViewProjection>>
          result = get_allowed_partial_start_invariant_machine_view_mappings(
              ms, shape, DeviceType::GPU);

      CHECK(correct == result);
    }

    SUBCASE("2 degrees of parallelism") {

      MachineSpecification ms = MachineSpecification(3, 3, 3, 0, 0);
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
        return StartInvariantMachineView{rect, DeviceType::GPU};
      };

      std::vector<StartInvariantMachineView> correct_mv = {
          make_2d_view(/*stride1*/ 1, /*stride2*/ 1),
          make_2d_view(/*stride1*/ 2, /*stride2*/ 1),
          make_2d_view(/*stride1*/ 1, /*stride2*/ 1),
          make_2d_view(/*stride1*/ 2, /*stride2*/ 1),
      };

      std::vector<MachineViewProjection> correct_proj = {
          MachineViewProjection{{{machine_view_dim_idx_t{0},
                                  MachineSpecificationDimension::INTRA},
                                 {machine_view_dim_idx_t{1},
                                  MachineSpecificationDimension::INTER}}},
          MachineViewProjection{{{machine_view_dim_idx_t{0},
                                  MachineSpecificationDimension::INTRA},
                                 {machine_view_dim_idx_t{1},
                                  MachineSpecificationDimension::INTER}}},

          MachineViewProjection{{{machine_view_dim_idx_t{0},
                                  MachineSpecificationDimension::INTER},
                                 {machine_view_dim_idx_t{1},
                                  MachineSpecificationDimension::INTRA}}},
          MachineViewProjection{{{machine_view_dim_idx_t{0},
                                  MachineSpecificationDimension::INTER},
                                 {machine_view_dim_idx_t{1},
                                  MachineSpecificationDimension::INTRA}}},
      };
      std::unordered_set<
          std::pair<StartInvariantMachineView, MachineViewProjection>>
          correct = unordered_set_of(zip(correct_mv, correct_proj));

      std::unordered_set<
          std::pair<StartInvariantMachineView, MachineViewProjection>>
          result = get_allowed_partial_start_invariant_machine_view_mappings(
              ms, shape, DeviceType::GPU);

      CHECK(result == correct);
    }
  }
}
