#include "op-attrs/ops/replicate.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Replicate shape inference") {
    ReplicateAttrs attrs = ReplicateAttrs{
        /*replicate_degree=*/4,
    };

    ParallelTensorShape input = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{10, 2},
                ShardParallelDim{12, 1},
                ShardParallelDim{14, 2},
                ShardParallelDim{16, 2},
            },
            ReplicaParallelDimSet{
                SumDegree{3},
                DiscardCopyDegree{2},
            },
        },
        DataType::FLOAT,
    };

    ParallelTensorShape result = get_output_shape(attrs, input);

    ParallelTensorShape correct_output = input;
    correct_output.dims.replica_dims.discard_copy_degree = DiscardCopyDegree{8};

    CHECK(result == correct_output);
  }
}
