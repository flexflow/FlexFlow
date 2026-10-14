#include "op-attrs/ops/combine.h"
#include "test/utils/doctest/fmt/expected.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Combine shape inference") {

    ParallelTensorShape input = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{12_p, 2_p},
                ShardParallelDim{14_p, 1_p},
                ShardParallelDim{16_p, 3_p},
                ShardParallelDim{18_p, 2_p},
            },
            ReplicaParallelDimSet{
                SumDegree{3_p},
                DiscardCopyDegree{2_p},
            },
        },
        DataType::FLOAT,
    };

    SUBCASE("valid") {
      ff_dim_t dim = ff_dim_t{2_n};
      int_ge_two degree = 3_ge2;
      CombineAttrs attrs = CombineAttrs{
          /*combine_dim=*/dim,
          /*combine_degree=*/degree,
      };

      ParallelTensorShape result =
          combine_get_output_parallel_shape(attrs, input);

      ParallelTensorShape correct = [&] {
        ParallelTensorShape output = input;
        positive_int old_shard_degree = output.dims.shard_dims.at(dim).degree;
        output.dims.shard_dims.at(dim).degree =
            positive_int{old_shard_degree / degree};
        return output;
      }();

      CHECK(result == correct);
    }

    SUBCASE("invalid") {
      ff_dim_t dim = ff_dim_t{2_n};
      int_ge_two degree = 4_ge2;
      CombineAttrs attrs = CombineAttrs{
          /*combine_dim=*/dim,
          /*combine_degree=*/degree,
      };

      CHECK_THROWS(combine_get_output_parallel_shape(attrs, input));
    }
  }
}
