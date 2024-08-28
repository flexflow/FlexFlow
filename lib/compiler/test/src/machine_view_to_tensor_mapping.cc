#include "compiler/machine_view_to_tensor_mapping.h"
#include "doctest/doctest.h"
#include "pcg/machine_view.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_set_of.h"

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_all_machine_view_to_tensor_mappings") {
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
    MachineView view = MachineView{
        device_id_from_index(0, DeviceType::GPU),
        StridedRectangle{{
            StridedRectangleSide{num_points_t(2), stride_t(1)},
            StridedRectangleSide{num_points_t(2), stride_t(4)},
            StridedRectangleSide{num_points_t(3), stride_t(1)},
        }},
    };

    bidict<machine_view_dim_idx_t, parallel_tensor_dim_idx_t> b1 = {
        {machine_view_dim_idx_t(2), parallel_tensor_dim_idx_t{ff_dim_t(0)}},
        {machine_view_dim_idx_t(1),
         parallel_tensor_dim_idx_t{ReplicaType::SUM}},
        {machine_view_dim_idx_t(0),
         parallel_tensor_dim_idx_t{ReplicaType::DISCARD_COPY}}};

    bidict<machine_view_dim_idx_t, parallel_tensor_dim_idx_t> b2 = {
        {machine_view_dim_idx_t(2), parallel_tensor_dim_idx_t{ff_dim_t(0)}},
        {machine_view_dim_idx_t(0),
         parallel_tensor_dim_idx_t{ReplicaType::SUM}},
        {machine_view_dim_idx_t(1),
         parallel_tensor_dim_idx_t{ReplicaType::DISCARD_COPY}}};

    std::unordered_set<MachineViewToTensorMapping> correct = {
        MachineViewToTensorMapping{b1}, MachineViewToTensorMapping{b2}};
    std::unordered_set<MachineViewToTensorMapping> result =
        get_all_machine_view_to_tensor_mappings(view, shape);

    CHECK(correct == result);
  }
}
