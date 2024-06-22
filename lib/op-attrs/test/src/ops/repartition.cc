#include "op-attrs/ops/repartition.h"
#include "test/utils/doctest.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Repartition shape inference") {
    ff_dim_t dim = ff_dim_t{2};
    int degree = 4;
    RepartitionAttrs attrs = RepartitionAttrs{
        /*repartition_dim=*/dim,
        /*repartition_degree=*/degree,
    };

    ParallelTensorShape input = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{12, 2},
                ShardParallelDim{14, 1},
                ShardParallelDim{16, 3},
                ShardParallelDim{18, 2},
            },
            ReplicaParallelDimSet{
                SumDegree{3},
                DiscardCopyDegree{2},
            },
        },
        DataType::FLOAT,
    };

    tl::expected<ParallelTensorShape, std::string> result =
        get_output_shape(attrs, input);

    tl::expected<ParallelTensorShape, std::string> correct = [&] {
      ParallelTensorShape output = input;
      output.dims.shard_dims.at(dim).degree *= degree;
      return output;
    }();

    CHECK(result == correct);
  }
}
