#include "op-attrs/ops/reshape.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("reshape_get_output_shape") {
    ReshapeAttrs attrs = ReshapeAttrs{
        TensorShape{
            TensorDims{FFOrdered<positive_int>{4_p, 3_p, 10_p}},
            DataType::FLOAT,
        },
    };

    SUBCASE("input tensor has different num elements") {
      TensorShape input = TensorShape{
          TensorDims{FFOrdered<positive_int>{4_p, 2_p, 10_p}},
          DataType::FLOAT,
      };

      CHECK_THROWS(reshape_get_output_shape(attrs, input));
    }

    SUBCASE("input tensor has different datatype") {
      TensorShape input = TensorShape{
          TensorDims{FFOrdered<positive_int>{4_p, 3_p, 10_p}},
          DataType::DOUBLE,
      };

      CHECK_THROWS(reshape_get_output_shape(attrs, input));
    }

    SUBCASE("valid input") {
      TensorShape input = TensorShape{
          TensorDims{FFOrdered<positive_int>{4_p, 6_p, 5_p}},
          DataType::FLOAT,
      };

      TensorShape result = reshape_get_output_shape(attrs, input);

      TensorShape correct = TensorShape{
          TensorDims{FFOrdered<positive_int>{4_p, 3_p, 10_p}},
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("reshape_get_output_parallel_dim_degrees") {
    ReshapeAttrs attrs = ReshapeAttrs{
        TensorShape{
            TensorDims{FFOrdered<positive_int>{4_p, 3_p, 10_p}},
            DataType::FLOAT,
        },
    };

    SUBCASE("input sum degree > 1") {
      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          SumDegree{3_p},
          DiscardCopyDegree{1_p},
          FFOrdered<positive_int>{1_p, 1_p, 1_p},
      };

      ParallelTensorDimDegrees result =
          reshape_get_output_parallel_dim_degrees(attrs, input);
      ParallelTensorDimDegrees correct = input;

      CHECK(result == correct);
    }

    SUBCASE("input discard copy degree > 1") {
      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          SumDegree{1_p},
          DiscardCopyDegree{2_p},
          FFOrdered<positive_int>{1_p, 1_p, 1_p},
      };

      ParallelTensorDimDegrees result =
          reshape_get_output_parallel_dim_degrees(attrs, input);
      ParallelTensorDimDegrees correct = input;

      CHECK(result == correct);
    }

    SUBCASE("does not allow shard degree") {
      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          SumDegree{1_p},
          DiscardCopyDegree{1_p},
          FFOrdered<positive_int>{1_p, 6_p, 1_p},
      };

      CHECK_THROWS(reshape_get_output_parallel_dim_degrees(attrs, input));
    }
  }
}
