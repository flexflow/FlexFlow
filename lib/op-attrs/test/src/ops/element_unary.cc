#include "op-attrs/ops/element_unary.h"
#include "op-attrs/parallel_tensor_shape.h"
#include <doctest/doctest.h>
#include "test/utils/doctest/fmt/expected.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("ReLU shape inference") {
    size_t d1 = 16;
    size_t d2 = 32;
    size_t d3 = 24;

    ElementUnaryAttrs attrs =
        ElementUnaryAttrs{OperatorType::RELU, std::nullopt};

    TensorShape input = TensorShape{
        TensorDims{
            FFOrdered<size_t>{
                d1,
                d2,
                d3,
            },
        },
        DataType::FLOAT,
    };

    tl::expected<TensorShape, std::string> result =
        get_output_shape(attrs, input);
    tl::expected<TensorShape, std::string> correct = input;

    CHECK(result == correct);

    auto make_i = [&](SumDegree o_sum,
                      DiscardCopyDegree o_eq,
                      int o_1,
                      int o_2,
                      int o_3) {
      return lift_to_parallel_with_degrees(
          input, o_sum, o_eq, FFOrdered<int>{o_1, o_2, o_3});
    };

    SUBCASE("partition i.e., sharding parallelism") {
      int degree1 = 4;
      int degree2 = 8;
      ParallelTensorShape par_input =
          make_i(SumDegree{1}, DiscardCopyDegree{1}, degree1, 1, degree2);

      tl::expected<ParallelTensorShape, std::string> result =
          get_output_shape(attrs, par_input);
      tl::expected<ParallelTensorShape, std::string> correct = par_input;

      CHECK(result == correct);
    }

    SUBCASE("sum degree > 1") {
      int degree = 2;

      tl::expected<ParallelTensorShape, std::string> result = get_output_shape(
          attrs, make_i(SumDegree{degree}, DiscardCopyDegree{1}, 1, 1, 1));

      CHECK_MESSAGE(!result.has_value(),
                    "Unexpected successful result: ",
                    result.error());
    }

    SUBCASE("discard copy degree > 1") {
      int degree = 2;

      tl::expected<ParallelTensorShape, std::string> result = get_output_shape(
          attrs, make_i(SumDegree{1}, DiscardCopyDegree{degree}, 1, 1, 1));

      CHECK_MESSAGE(!result.has_value(),
                    "Unexpected successful result: ",
                    result.error());
    }
  }
}
