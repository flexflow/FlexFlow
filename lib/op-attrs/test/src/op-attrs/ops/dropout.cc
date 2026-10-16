#include "op-attrs/ops/dropout.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "test/utils/doctest/fmt/optional.h"
#include "utils/expected.h"
#include "utils/fmt/expected.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_output_shape(DropoutAttrs, TensorShape)") {
    DropoutAttrs attrs = DropoutAttrs{
        /*rate=*/0.5,
        /*seed=*/1,
    };

    TensorShape input = TensorShape{
        TensorDims{FFOrdered{
            12_p,
            14_p,
            16_p,
        }},
        DataType::FLOAT,
    };

    TensorShape result = get_output_shape(attrs, input);
    TensorShape correct = input;

    CHECK(result == correct);
  }

  TEST_CASE("get_output_shape(DropoutAttrs, ParallelTensorShape)") {
    DropoutAttrs attrs = DropoutAttrs{
        /*rate=*/0.5,
        /*seed=*/1,
    };

    TensorShape input = TensorShape{
        TensorDims{FFOrdered{
            12_p,
            14_p,
            16_p,
        }},
        DataType::FLOAT,
    };

    TensorShape output = input;

    auto make_input = [&](SumDegree o_sum,
                          DiscardCopyDegree o_eq,
                          positive_int o0,
                          positive_int o1,
                          positive_int o2) {
      return lift_shape_to_parallel_with_degrees(
          input, o_sum, o_eq, FFOrdered{o0, o1, o2});
    };

    auto make_output = [&](SumDegree o_sum,
                           DiscardCopyDegree o_eq,
                           positive_int o0,
                           positive_int o1,
                           positive_int o2) {
      return lift_shape_to_parallel_with_degrees(
          output, o_sum, o_eq, FFOrdered{o0, o1, o2});
    };

    SUBCASE("partition parallelism (allowed)") {
      positive_int degree0 = 2_p;
      positive_int degree2 = 4_p;

      ParallelTensorShape par_input = make_input(
          SumDegree{1_p}, DiscardCopyDegree{1_p}, degree0, 1_p, degree2);

      ParallelTensorShape result = get_output_shape(attrs, par_input);
      ParallelTensorShape correct = make_output(
          SumDegree{1_p}, DiscardCopyDegree{1_p}, degree0, 1_p, degree2);

      CHECK(result == correct);
    }

    SUBCASE("sum parallelism (not allowed)") {
      SumDegree sum_degree = SumDegree{2_p};

      ParallelTensorShape par_input =
          make_input(sum_degree, DiscardCopyDegree{1_p}, 1_p, 1_p, 1_p);

      CHECK_THROWS(get_output_shape(attrs, par_input));
    }

    SUBCASE("discard copy parallelism (not allowed)") {
      DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{2_p};

      ParallelTensorShape par_input =
          make_input(SumDegree{1_p}, discard_copy_degree, 1_p, 1_p, 1_p);

      CHECK_THROWS(get_output_shape(attrs, par_input));
    }
  }
}
