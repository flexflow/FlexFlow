#include "op-attrs/ops/reduction.h"
#include <doctest/doctest.h>
#include "test/utils/doctest/fmt/expected.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Reduction shape inference") {

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
      int degree = 3;
      ReductionAttrs attrs = ReductionAttrs{
          /*repartition_degree=*/degree,
      };

      tl::expected<ParallelTensorShape, std::string> result =
          get_output_shape(attrs, input);

      tl::expected<ParallelTensorShape, std::string> correct = [&] {
        ParallelTensorShape output = input;
        output.dims.replica_dims.sum_degree.value /= degree;
        return output;
      }();

      CHECK(result == correct);
    }

    SUBCASE("invalid") {
      int degree = 4;
      ReductionAttrs attrs = ReductionAttrs{
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
