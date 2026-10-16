#include "op-attrs/ops/split.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("split_get_output_shapes") {
    TensorShape input_shape = TensorShape{
        TensorDims{FFOrdered{4_p, 3_p, 10_p, 10_p}},
        DataType::FLOAT,
    };

    SUBCASE("splits are too small for input") {
      SplitAttrs attrs = SplitAttrs{
          /*splits=*/std::vector<positive_int>{
              2_p,
              4_p,
              3_p,
          },
          /*axis=*/ff_dim_t{2_n},
      };

      CHECK_THROWS(split_get_output_shapes(attrs, input_shape));
    }

    SUBCASE("splits are too large for input") {
      SplitAttrs attrs = SplitAttrs{
          /*splits=*/std::vector<positive_int>{
              2_p,
              6_p,
              3_p,
          },
          /*axis=*/ff_dim_t{2_n},
      };

      CHECK_THROWS(split_get_output_shapes(attrs, input_shape));
    }

    SUBCASE("axis does not exist in input") {
      SplitAttrs attrs = SplitAttrs{
          /*splits=*/std::vector<positive_int>{
              2_p,
              5_p,
              3_p,
          },
          /*axis=*/ff_dim_t{4_n},
      };

      CHECK_THROWS(split_get_output_shapes(attrs, input_shape));
    }

    SUBCASE("correct usage") {
      SplitAttrs attrs = SplitAttrs{
          /*splits=*/std::vector<positive_int>{
              2_p,
              5_p,
              3_p,
          },
          /*axis=*/ff_dim_t{2_n},
      };

      std::vector<TensorShape> result =
          split_get_output_shapes(attrs, input_shape);

      auto mk_correct_shape = [&](positive_int x) -> TensorShape {
        return TensorShape{
            TensorDims{FFOrdered{4_p, 3_p, x, 10_p}},
            DataType::FLOAT,
        };
      };

      std::vector<TensorShape> correct = {
          mk_correct_shape(2_p),
          mk_correct_shape(5_p),
          mk_correct_shape(3_p),
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("split_get_output_parallel_dim_degrees") {
    SplitAttrs attrs = SplitAttrs{
        /*splits=*/std::vector<positive_int>{
            3_p,
            2_p,
            5_p,
        },
        /*axis=*/ff_dim_t{2_n},
    };

    SUBCASE("split degree is 1") {
      ParallelTensorDimDegrees input_dim_degrees = ParallelTensorDimDegrees{
          /*sum_degree=*/SumDegree{2_p},
          /*discard_copy_degree=*/DiscardCopyDegree{1_p},
          /*shard_degrees=*/
          FFOrdered<positive_int>{
              2_p,
              1_p,
              1_p,
              1_p,
          },
      };

      std::vector<ParallelTensorDimDegrees> result =
          split_get_output_parallel_dim_degrees(attrs, input_dim_degrees);
      std::vector<ParallelTensorDimDegrees> correct = {
          input_dim_degrees,
          input_dim_degrees,
          input_dim_degrees,
      };

      CHECK(result == correct);
    }

    SUBCASE("split degree is not 1") {
      ParallelTensorDimDegrees input_dim_degrees = ParallelTensorDimDegrees{
          /*sum_degree=*/SumDegree{1_p},
          /*discard_copy_degree=*/DiscardCopyDegree{1_p},
          /*shard_degrees=*/
          FFOrdered<positive_int>{
              1_p,
              1_p,
              2_p,
              1_p,
          },
      };

      CHECK_THROWS(
          split_get_output_parallel_dim_degrees(attrs, input_dim_degrees));
    }
  }
}
