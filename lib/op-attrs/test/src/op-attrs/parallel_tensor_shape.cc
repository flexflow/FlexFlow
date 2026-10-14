#include "op-attrs/parallel_tensor_shape.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_piece_shape") {
    ParallelTensorShape input = ParallelTensorShape{
        /*dims=*/ParallelTensorDims{
            /*shard_dims=*/FFOrdered<ShardParallelDim>{
                ShardParallelDim{9_p, 1_p},
                ShardParallelDim{12_p, 4_p},
                ShardParallelDim{8_p, 1_p},
            },
            /*replica_dims=*/
            ReplicaParallelDimSet{
                /*sum_degree=*/SumDegree{2_p},
                /*discard_copy_degree=*/DiscardCopyDegree{3_p},
            },
        },
        /*data_type=*/DataType::FLOAT,
    };

    TensorShape result = get_piece_shape(input);
    TensorShape correct = TensorShape{
        /*dims=*/TensorDims{
            FFOrdered<positive_int>{
                9_p,
                3_p,
                8_p,
            },
        },
        /*data_type=*/DataType::FLOAT,
    };

    CHECK(result == correct);
  }

  TEST_CASE("get_reduced_shape") {
    ParallelTensorShape input = ParallelTensorShape{
        /*dims=*/ParallelTensorDims{
            /*shard_dims=*/FFOrdered<ShardParallelDim>{
                ShardParallelDim{9_p, 1_p},
                ShardParallelDim{12_p, 4_p},
                ShardParallelDim{8_p, 1_p},
            },
            /*replica_dims=*/
            ReplicaParallelDimSet{
                /*sum_degree=*/SumDegree{2_p},
                /*discard_copy_degree=*/DiscardCopyDegree{3_p},
            },
        },
        /*data_type=*/DataType::FLOAT,
    };

    TensorShape result = get_reduced_shape(input);
    TensorShape correct = TensorShape{
        /*dims=*/TensorDims{
            FFOrdered<positive_int>{
                9_p,
                12_p,
                8_p,
            },
        },
        /*data_type=*/DataType::FLOAT,
    };

    CHECK(result == correct);
  }
}
