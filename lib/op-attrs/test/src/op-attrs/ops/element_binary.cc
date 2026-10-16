#include "op-attrs/ops/element_binary.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_dims.h"
#include "test/utils/doctest/fmt/expected.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("EWAdd shape inference") {
    positive_int d1 = 16_p;
    positive_int d2 = 32_p;
    positive_int d3 = 24_p;

    ElementBinaryAttrs attrs = ElementBinaryAttrs{
        ElementBinaryOp::ADD,
        DataType::FLOAT,
        /*should_broadcast_lhs=*/false,
        /*should_broadcast_rhs=*/false,
    };

    TensorShape input_lhs = TensorShape{
        TensorDims{
            FFOrdered{
                d1,
                d2,
                d3,
            },
        },
        DataType::FLOAT,
    };

    TensorShape input_rhs = input_lhs;

    SUBCASE("correct") {
      TensorShape result = get_output_shape(attrs, input_lhs, input_rhs);
      TensorShape correct = input_lhs;

      CHECK(result == correct);
    }

    SUBCASE("mismatched dim size") {
      TensorShape incorrect_rhs = input_lhs;
      dim_at_idx(incorrect_rhs.dims, relative_ff_dim_t{0}) += 1_p;

      CHECK_THROWS(get_output_shape(attrs, input_lhs, incorrect_rhs));
    }
  }

  TEST_CASE("EWAdd parallel shape inference") {
    positive_int d1 = 16_p;
    positive_int d2 = 32_p;
    positive_int d3 = 24_p;

    ElementBinaryAttrs attrs = ElementBinaryAttrs{
        ElementBinaryOp::ADD,
        DataType::FLOAT,
        /*should_broadcast_lhs=*/false,
        /*should_broadcast_rhs=*/false,
    };

    TensorShape unpar_lhs = TensorShape{
        TensorDims{
            FFOrdered{
                d1,
                d2,
                d3,
            },
        },
        DataType::FLOAT,
    };

    TensorShape unpar_rhs = unpar_lhs;
    TensorShape unpar_output = get_output_shape(attrs, unpar_lhs, unpar_rhs);

    auto make_lhs = [&](SumDegree o_sum,
                        DiscardCopyDegree o_eq,
                        positive_int o_1,
                        positive_int o_2,
                        positive_int o_3) {
      return lift_shape_to_parallel_with_degrees(
          unpar_lhs, o_sum, o_eq, FFOrdered{o_1, o_2, o_3});
    };

    auto make_rhs = [&](SumDegree o_sum,
                        DiscardCopyDegree o_eq,
                        positive_int o_1,
                        positive_int o_2,
                        positive_int o_3) {
      return lift_shape_to_parallel_with_degrees(
          unpar_rhs, o_sum, o_eq, FFOrdered{o_1, o_2, o_3});
    };

    auto make_output = [&](SumDegree o_sum,
                           DiscardCopyDegree o_eq,
                           positive_int o_1,
                           positive_int o_2,
                           positive_int o_3) {
      return lift_shape_to_parallel_with_degrees(
          unpar_output, o_sum, o_eq, FFOrdered{o_1, o_2, o_3});
    };

    SUBCASE("data parallelism") {
      positive_int degree = 4_p;

      ParallelTensorShape input_lhs =
          make_lhs(SumDegree{1_p}, DiscardCopyDegree{1_p}, degree, 1_p, 1_p);
      ParallelTensorShape input_rhs =
          make_rhs(SumDegree{1_p}, DiscardCopyDegree{1_p}, degree, 1_p, 1_p);
      ParallelTensorShape result =
          get_output_shape(attrs, input_lhs, input_rhs);
      ParallelTensorShape correct =
          make_output(SumDegree{1_p}, DiscardCopyDegree{1_p}, degree, 1_p, 1_p);

      CHECK(result == correct);
    }

    SUBCASE("reduction parallelism") {
      positive_int degree = 4_p;

      ParallelTensorShape input_lhs =
          make_lhs(SumDegree{degree}, DiscardCopyDegree{1_p}, 1_p, 1_p, 1_p);
      ParallelTensorShape input_rhs =
          make_rhs(SumDegree{degree}, DiscardCopyDegree{1_p}, 1_p, 1_p, 1_p);
      ParallelTensorShape result =
          get_output_shape(attrs, input_lhs, input_rhs);
      ParallelTensorShape correct =
          make_output(SumDegree{degree}, DiscardCopyDegree{1_p}, 1_p, 1_p, 1_p);

      CHECK(result == correct);
    }

    SUBCASE("invalid discard copy parallelism") {
      positive_int degree = 4_p;

      ParallelTensorShape input_lhs =
          make_lhs(SumDegree{1_p}, DiscardCopyDegree{degree}, 1_p, 1_p, 1_p);
      ParallelTensorShape input_rhs =
          make_rhs(SumDegree{1_p}, DiscardCopyDegree{degree}, 1_p, 1_p, 1_p);

      CHECK_THROWS(get_output_shape(attrs, input_lhs, input_rhs));
    }

    SUBCASE("invalid mismatched parallelism degrees") {
      positive_int degree = 4_p;

      ParallelTensorShape input_lhs =
          make_lhs(SumDegree{1_p}, DiscardCopyDegree{1_p}, 1_p, degree, 1_p);
      ParallelTensorShape input_rhs =
          make_rhs(SumDegree{1_p}, DiscardCopyDegree{1_p}, 1_p, 1_p, degree);

      CHECK_THROWS(get_output_shape(attrs, input_lhs, input_rhs));
    }
  }
}
