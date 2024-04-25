#include "doctest/doctest.h"
#include "op-attrs/ops/conv_2d.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_output_shape(Conv2DAttrs, TensorShape)") {
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

    Conv2DAttrs attrs = {
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
    size_t input_channels = 6;
    size_t input_height = 10;
    size_t input_width = 15;

    TensorShape input_shape = {
      TensorDims{
        FFOrdered<size_t>{
          num_samples,
          input_channels,
          input_height,
          input_width,
        }
      },
      DataType::FLOAT,
    };

    TensorShape result = get_output_shape(attrs, input_shape);


    size_t correct_output_height = 3;
    size_t correct_output_width = 6;

    TensorShape correct_output_shape = {
      TensorDims{
        FFOrdered<size_t>{
          num_samples,     
          static_cast<size_t>(out_channels), 
          correct_output_height,
          correct_output_width,
        }
      },
      DataType::FLOAT,
    };

    CHECK(result == correct_output_shape);
  }
}
