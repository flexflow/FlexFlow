#include "op-attrs/ops/combine.h"
#include <doctest/doctest.h>
#include "test/utils/doctest/fmt/expected.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Combine shape inference") {

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

    SUBCASE("valid") {
      ff_dim_t dim = ff_dim_t{2};
      int degree = 3;
      CombineAttrs attrs = CombineAttrs{
          /*repartition_dim=*/dim,
          /*repartition_degree=*/degree,
      };

      tl::expected<ParallelTensorShape, std::string> result =
          get_output_shape(attrs, input);

      tl::expected<ParallelTensorShape, std::string> correct = [&] {
        ParallelTensorShape output = input;
        output.dims.shard_dims.at(dim).degree /= degree;
        return output;
      }();

      CHECK(result == correct);
    }

    SUBCASE("invalid") {
      ff_dim_t dim = ff_dim_t{2};
      int degree = 4;
      CombineAttrs attrs = CombineAttrs{
          /*repartition_dim=*/dim,
          /*repartition_degree=*/degree,
      };

      tl::expected<ParallelTensorShape, std::string> result =
          get_output_shape(attrs, input);

      CHECK_MESSAGE(!result.has_value(),
                    "Unexpected successful result: ",
                    result.error());
    }
  }
}
