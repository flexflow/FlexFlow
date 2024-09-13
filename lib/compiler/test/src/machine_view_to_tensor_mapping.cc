#include "compiler/machine_view_to_tensor_mapping.h"
#include "doctest/doctest.h"
#include "pcg/machine_view.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/fmt/unordered_map.h"
#include "utils/fmt/unordered_set.h"

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_all_machine_view_to_tensor_mappings") {
    SUBCASE("no possible mappings") {
      ParallelTensorShape shape = ParallelTensorShape{
          ParallelTensorDims{
              FFOrdered<ShardParallelDim>{
                  ShardParallelDim{3, 1},
              },
              ReplicaParallelDimSet{
                  SumDegree{1},
                  DiscardCopyDegree{2},
              },
          },
          DataType::FLOAT,
      };
      MachineView view =
          MachineView{MachineViewCoordinate{{0, 0, 0}},
                      StridedRectangle{{
                          StridedRectangleSide{num_points_t{2}, stride_t{1}},
                          StridedRectangleSide{num_points_t{2}, stride_t{4}},
                      }},
                      DeviceType::GPU};
      CHECK_THROWS_AS(get_all_machine_view_to_tensor_mappings(view, shape),
                      std::runtime_error);
    }
    SUBCASE("multiple possible mappings") {
      ParallelTensorShape shape = ParallelTensorShape{
          ParallelTensorDims{
              FFOrdered<ShardParallelDim>{
                  ShardParallelDim{10, 3},
              },
              ReplicaParallelDimSet{
                  SumDegree{2},
                  DiscardCopyDegree{2},
              },
          },
          DataType::FLOAT,
      };
      MachineView view =
          MachineView{MachineViewCoordinate{{0, 0, 0}},
                      StridedRectangle{{
                          StridedRectangleSide{num_points_t{2}, stride_t{1}},
                          StridedRectangleSide{num_points_t{2}, stride_t{4}},
                          StridedRectangleSide{num_points_t{3}, stride_t{1}},
                      }},
                      DeviceType::GPU};

      machine_view_dim_idx_t mv_dim_0 = machine_view_dim_idx_t{0};
      machine_view_dim_idx_t mv_dim_1 = machine_view_dim_idx_t{1};
      machine_view_dim_idx_t mv_dim_2 = machine_view_dim_idx_t{2};
      parallel_tensor_dim_idx_t pt_dim_0 =
          parallel_tensor_dim_idx_t{ff_dim_t{0}};
      parallel_tensor_dim_idx_t pt_dim_sum =
          parallel_tensor_dim_idx_t{ReplicaType::SUM};
      parallel_tensor_dim_idx_t pt_dim_eq =
          parallel_tensor_dim_idx_t{ReplicaType::DISCARD_COPY};

      bidict<machine_view_dim_idx_t, parallel_tensor_dim_idx_t> b1 = {
          {mv_dim_2, pt_dim_0},
          {mv_dim_1, pt_dim_sum},
          {mv_dim_0, pt_dim_eq},
      };

      bidict<machine_view_dim_idx_t, parallel_tensor_dim_idx_t> b2 = {
          {mv_dim_2, pt_dim_0},
          {mv_dim_0, pt_dim_sum},
          {mv_dim_1, pt_dim_eq},
      };

      std::unordered_set<MachineViewToTensorMapping> correct = {
          MachineViewToTensorMapping{b1}, MachineViewToTensorMapping{b2}};
      std::unordered_set<MachineViewToTensorMapping> result =
          get_all_machine_view_to_tensor_mappings(view, shape);

      CHECK(correct == result);
    }
  }
}
