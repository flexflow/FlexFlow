#include <doctest/doctest.h>
#include "op-attrs/ops/softmax.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/expected.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_output_shape(SoftmaxAttrs, TensorShape)") {
    TensorShape input = TensorShape{
      TensorDims{FFOrdered<size_t>{
        12,
        14,
        16,
      }},
      DataType::FLOAT,
    };

    SUBCASE("attrs.dim in bounds") {
      SoftmaxAttrs attrs = SoftmaxAttrs{ff_dim_t{1}};  

      tl::expected<TensorShape, std::string> result = get_output_shape(attrs, input);
      tl::expected<TensorShape, std::string> correct = input;

      CHECK(result == correct);
    }

    SUBCASE("attrs.dims out of bounds") {
      SoftmaxAttrs attrs = SoftmaxAttrs{ff_dim_t{4}};  

      std::optional<TensorShape> result = optional_from_expected(get_output_shape(attrs, input));
      std::optional<TensorShape> correct = std::nullopt;

      CHECK(result == correct);
    }
  }

  TEST_CASE("get_output_shape(SoftmaxAttrs, ParallelTensorShape)") {
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

    SUBCASE("partition parallelism in non-softmax-dim (valid)") {
      int degree0 = 2;
      int degree2 = 4;

      ParallelTensorShape par_input = make_input(SumDegree{1}, DiscardCopyDegree{1}, degree0, 1, degree2);

      SUBCASE("attrs.dim in bounds") {
        SoftmaxAttrs attrs = SoftmaxAttrs{ff_dim_t{1}};  

        tl::expected<ParallelTensorShape, std::string> result = get_output_shape(attrs, par_input);
        tl::expected<ParallelTensorShape, std::string> correct = make_output(SumDegree{1}, DiscardCopyDegree{1}, degree0, 1, degree2);

        CHECK(result == correct);
      }

      SUBCASE("attrs.dims out of bounds") {
        SoftmaxAttrs attrs = SoftmaxAttrs{ff_dim_t{4}};  

        std::optional<ParallelTensorShape> result = optional_from_expected(get_output_shape(attrs, par_input));
        std::optional<ParallelTensorShape> correct = std::nullopt;

        CHECK(result == correct);
      }
    }

    SUBCASE("partition parallism in softmax dim (invalid)") {
      int degree1 = 2;

      SoftmaxAttrs attrs = SoftmaxAttrs{ff_dim_t{1}};  

      ParallelTensorShape par_input = make_input(SumDegree{1}, DiscardCopyDegree{1}, 1, degree1, 1);

      std::optional<ParallelTensorShape> result = optional_from_expected(get_output_shape(attrs, par_input));
      std::optional<ParallelTensorShape> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("sum parallelism (invalid)") {
      SumDegree sum_degree = SumDegree{2};

      SoftmaxAttrs attrs = SoftmaxAttrs{ff_dim_t{1}};  

      ParallelTensorShape par_input = make_input(sum_degree, DiscardCopyDegree{1}, 1, 1, 1);

      std::optional<ParallelTensorShape> result = optional_from_expected(get_output_shape(attrs, par_input));
      std::optional<ParallelTensorShape> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("discard copy parallelism (invalid)") {
      DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{2};

      SoftmaxAttrs attrs = SoftmaxAttrs{ff_dim_t{1}};  

      ParallelTensorShape par_input = make_input(SumDegree{1}, discard_copy_degree, 1, 1, 1);

      std::optional<ParallelTensorShape> result = optional_from_expected(get_output_shape(attrs, par_input));
      std::optional<ParallelTensorShape> correct = std::nullopt;

      CHECK(result == correct);
    }
  }
}
