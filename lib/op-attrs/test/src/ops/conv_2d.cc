#include "op-attrs/ops/conv_2d.h"
#include "doctest/doctest.h"
#include "utils/integer_conversions.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Conv2D shape inference") {
    int out_channels = 4;
    int kernel_h = 3;
    int kernel_w = 2;
    int stride_h = 2;
    int stride_w = 2;
    int padding_h = 1;
    int padding_w = 1;
    int groups = 1;
    std::optional<Activation> activation = std::nullopt;
    bool use_bias = true;

    Conv2DAttrs attrs = Conv2DAttrs{
        /*out_channels=*/out_channels,
        /*kernel_h=*/kernel_h,
        /*kernel_w=*/kernel_w,
        /*stride_h=*/stride_h,
        /*stride_w=*/stride_w,
        /*padding_h=*/padding_h,
        /*padding_w=*/padding_w,
        /*groups=*/groups,
        /*activation=*/activation,
        /*use_bias=*/true,
    };

    size_t num_samples = 7;
    size_t input_channels = 4;
    size_t input_height = 11;
    size_t input_width = 15;

    TensorShape input = TensorShape{
        TensorDims{FFOrdered<size_t>{
            num_samples,
            input_channels,
            input_height,
            input_width,
        }},
        DataType::FLOAT,
    };

    size_t output_height = 6;
    size_t output_width = 8;

    TensorShape output = TensorShape{
        TensorDims{FFOrdered<size_t>{
            num_samples,
            size_t_from_int(out_channels),
            output_height,
            output_width,
        }},
        DataType::FLOAT,
    };

    TensorShape kernel = TensorShape{
        TensorDims{FFOrdered<size_t>{
            size_t_from_int(out_channels),
            input_channels,
            size_t_from_int(kernel_h),
            size_t_from_int(kernel_w),
        }},
        DataType::FLOAT,
    };

    TensorShape bias = TensorShape{
        TensorDims{FFOrdered<size_t>{
            size_t_from_int(out_channels),
        }},
        DataType::FLOAT,
    };

    SUBCASE("get_output_shape(Conv2DAttrs, TensorShape)") {
      TensorShape result_output = get_output_shape(attrs, input);
      TensorShape correct_output = output;
      CHECK(result_output == correct_output);
    }

    SUBCASE("get_kernel_shape(Conv2DAttrs, TensorShape)") {
      TensorShape result_kernel = get_kernel_shape(attrs, input);
      TensorShape correct_kernel = kernel;
      CHECK(result_kernel == correct_kernel);
    }

    SUBCASE("get_bias_shape(Conv2DAttrs, TensorShape)") {
      TensorShape result_bias = get_bias_shape(attrs, input);
      TensorShape correct_bias = bias;
      CHECK(result_bias == correct_bias);
    }

    auto make_input = [&](SumDegree o_sum,
                          DiscardCopyDegree o_eq,
                          int o_n,
                          int o_c,
                          int o_h,
                          int o_w) {
      return lift_to_parallel_with_degrees(
          input, o_sum, o_eq, FFOrdered<int>{o_n, o_c, o_h, o_w});
    };

    auto make_output = [&](SumDegree o_sum,
                           DiscardCopyDegree o_eq,
                           int o_n,
                           int o_c,
                           int o_h,
                           int o_w) {
      return lift_to_parallel_with_degrees(
          output, o_sum, o_eq, FFOrdered<int>{o_n, o_c, o_h, o_w});
    };

    auto make_kernel = [&](SumDegree o_sum,
                           DiscardCopyDegree o_eq,
                           int o_outchannels,
                           int o_inchannels,
                           int o_kernel_h,
                           int o_kernel_w) {
      return lift_to_parallel_with_degrees(
          kernel,
          o_sum,
          o_eq,
          FFOrdered<int>{o_outchannels, o_inchannels, o_kernel_h, o_kernel_w});
    };

    auto make_bias =
        [&](SumDegree o_sum, DiscardCopyDegree o_eq, int o_outchannels) {
          return lift_to_parallel_with_degrees(
              bias, o_sum, o_eq, FFOrdered<int>{o_outchannels});
        };

    SUBCASE("data parallelism") {
      int degree = 2;
      ParallelTensorShape par_input =
          make_input(SumDegree{1}, DiscardCopyDegree{1}, degree, 1, 1, 1);

      SUBCASE("get_output_shape") {
        ParallelTensorShape result = get_output_shape(attrs, par_input);
        ParallelTensorShape correct =
            make_output(SumDegree{1}, DiscardCopyDegree{1}, degree, 1, 1, 1);
        CHECK(result == correct);
      }

      SUBCASE("get_kernel_shape") {
        ParallelTensorShape result = get_kernel_shape(attrs, par_input);
        ParallelTensorShape correct =
            make_kernel(SumDegree{1}, DiscardCopyDegree{degree}, 1, 1, 1, 1);
        CHECK(result == correct);
      }

      SUBCASE("get_bias_shape") {
        ParallelTensorShape result = get_bias_shape(attrs, par_input);
        ParallelTensorShape correct =
            make_bias(SumDegree{1}, DiscardCopyDegree{degree}, 1);
        CHECK(result == correct);
      }
    }

    SUBCASE("input channel parallelism") {
      int degree = 2;
      ParallelTensorShape par_input =
          make_input(SumDegree{1}, DiscardCopyDegree{1}, 1, degree, 1, 1);

      SUBCASE("get_output_shape") {
        ParallelTensorShape result = get_output_shape(attrs, par_input);
        ParallelTensorShape correct =
            make_output(SumDegree{degree}, DiscardCopyDegree{1}, 1, 1, 1, 1);
        CHECK(result == correct);
      }

      SUBCASE("get_kernel_shape") {
        ParallelTensorShape result = get_kernel_shape(attrs, par_input);
        ParallelTensorShape correct =
            make_kernel(SumDegree{1}, DiscardCopyDegree{1}, 1, degree, 1, 1);
        CHECK(result == correct);
      }

      SUBCASE("get_bias_shape") {
        ParallelTensorShape result = get_bias_shape(attrs, par_input);
        ParallelTensorShape correct =
            make_bias(SumDegree{degree}, DiscardCopyDegree{1}, 1);
        CHECK(result == correct);
      }
    }

    SUBCASE("output channel parallelism") {
      int degree = 2;
      ParallelTensorShape par_input =
          make_input(SumDegree{1}, DiscardCopyDegree{degree}, 1, 1, 1, 1);

      SUBCASE("get_output_shape") {
        ParallelTensorShape result = get_output_shape(attrs, par_input);
        ParallelTensorShape correct =
            make_output(SumDegree{1}, DiscardCopyDegree{1}, 1, degree, 1, 1);
        CHECK(result == correct);
      }

      SUBCASE("get_kernel_shape") {
        ParallelTensorShape result = get_kernel_shape(attrs, par_input);
        ParallelTensorShape correct =
            make_kernel(SumDegree{1}, DiscardCopyDegree{1}, degree, 1, 1, 1);
        CHECK(result == correct);
      }

      SUBCASE("get_bias_shape") {
        ParallelTensorShape result = get_bias_shape(attrs, par_input);
        ParallelTensorShape correct =
            make_bias(SumDegree{1}, DiscardCopyDegree{1}, degree);
        CHECK(result == correct);
      }
    }

    SUBCASE("propagating sum degree") {
      int degree = 2;
      ParallelTensorShape par_input =
          make_input(SumDegree{degree}, DiscardCopyDegree{1}, 1, 1, 1, 1);

      SUBCASE("get_output_shape") {
        ParallelTensorShape result = get_output_shape(attrs, par_input);
        ParallelTensorShape correct =
            make_output(SumDegree{degree}, DiscardCopyDegree{1}, 1, 1, 1, 1);
        CHECK(result == correct);
      }

      SUBCASE("get_kernel_shape") {
        ParallelTensorShape result = get_kernel_shape(attrs, par_input);
        ParallelTensorShape correct =
            make_kernel(SumDegree{1}, DiscardCopyDegree{degree}, 1, 1, 1, 1);
        CHECK(result == correct);
      }

      SUBCASE("get_bias_shape") {
        ParallelTensorShape result = get_bias_shape(attrs, par_input);
        ParallelTensorShape correct =
            make_bias(SumDegree{degree}, DiscardCopyDegree{1}, 1);
        CHECK(result == correct);
      }
    }
  }
}
