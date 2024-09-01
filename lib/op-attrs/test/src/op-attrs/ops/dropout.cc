#include <doctest/doctest.h>
#include "op-attrs/ops/dropout.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/expected.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_output_shape(DropoutAttrs, TensorShape)") {
    DropoutAttrs attrs = DropoutAttrs{
      /*rate=*/0.5,
      /*seed=*/1,
    };

    TensorShape input = TensorShape{
      TensorDims{FFOrdered<size_t>{
        12,
        14,
        16,
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
      TensorDims{FFOrdered<size_t>{
        12,
        14,
        16,
      }},
      DataType::FLOAT,
    };

    TensorShape output = input;

    auto make_input = [&](SumDegree o_sum,
                          DiscardCopyDegree o_eq,
                          int o0,
                          int o1,
                          int o2) {
      return lift_to_parallel_with_degrees(
          input, o_sum, o_eq, FFOrdered<int>{o0, o1, o2});
    };

    auto make_output = [&](SumDegree o_sum,
                           DiscardCopyDegree o_eq,
                           int o0,
                           int o1,
                           int o2) {
      return lift_to_parallel_with_degrees(
          output, o_sum, o_eq, FFOrdered<int>{o0, o1, o2});
    };

    SUBCASE("partition parallelism (allowed)") {
        int degree0 = 2;
        int degree2 = 4;

        ParallelTensorShape par_input = make_input(SumDegree{1}, DiscardCopyDegree{1}, degree0, 1, degree2);

        tl::expected<ParallelTensorShape, std::string> result = get_output_shape(attrs, par_input);
        tl::expected<ParallelTensorShape, std::string> correct = make_output(SumDegree{1}, DiscardCopyDegree{1}, degree0, 1, degree2);

        CHECK(result == correct);
    }

    SUBCASE("sum parallelism (not allowed)") {
      SumDegree sum_degree = SumDegree{2};

      ParallelTensorShape par_input = make_input(sum_degree, DiscardCopyDegree{1}, 1, 1, 1);

      std::optional<ParallelTensorShape> result = optional_from_expected(get_output_shape(attrs, par_input));
      std::optional<ParallelTensorShape> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("discard copy parallelism (not allowed)") {
      DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{2};

      ParallelTensorShape par_input = make_input(SumDegree{1}, discard_copy_degree, 1, 1, 1);

      std::optional<ParallelTensorShape> result = optional_from_expected(get_output_shape(attrs, par_input));
      std::optional<ParallelTensorShape> correct = std::nullopt;

      CHECK(result == correct);
    }
  }
}
