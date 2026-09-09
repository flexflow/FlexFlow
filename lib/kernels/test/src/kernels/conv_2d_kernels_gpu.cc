#include "kernels/conv_2d_kernels_gpu.h"
#include "internal/test_utils.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/format_accessor_contents.h"
#include "test/utils/doctest/check_kv.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

static Conv2DAttrs make_attrs() {
  return Conv2DAttrs{
      /*out_channels=*/3_p,
      /*kernel_h=*/3_p,
      /*kernel_w=*/3_p,
      /*stride_h=*/1_p,
      /*stride_w=*/1_p,
      /*padding_h=*/1_n,
      /*padding_w=*/1_n,
      /*groups=*/1_p,
      /*activation=*/std::nullopt,
      /*use_bias=*/true,
  };
}

// NCHW
static GenericTensorAccessorR make_input(Allocator &allocator) {
  return create_4d_accessor_r_with_contents<float>(
      {
          {
              {
                  {1, 2, 3, 4},
                  {0.5, 0.25, -1, -2},
                  {-3, 0.125, 2, 1},
                  {4, -0.5, 0.75, 0},
              },
              {
                  {-1, 0.5, 2, -2},
                  {3, 1, 0.25, 0.5},
                  {-0.25, -1.5, 1.25, 2.5},
                  {0.75, 3, -4, 1},
              },
          },
      },
      allocator);
}

static GenericTensorAccessorR make_filter(Allocator &allocator) {
  return create_4d_accessor_r_with_contents<float>(
      {
          {
              {{1, 0, -1}, {0.5, 1, 0.5}, {-1, 0, 1}},
              {{0.25, -0.5, 0.75}, {1, -1, 0.5}, {0, 0.25, -0.25}},
          },
          {
              {{-1, 2, 0.5}, {0, 1, -1}, {0.25, 0.5, 0.75}},
              {{1, 1, 1}, {-0.5, -0.5, -0.5}, {0.25, 0, -0.25}},
          },
          {
              {{0.5, 0.5, 0.5}, {0.25, 0.25, 0.25}, {-1, -1, -1}},
              {{2, -1, 0}, {0.5, 0.75, -0.25}, {1, 0.5, 0.25}},
          },
      },
      allocator);
}

static GenericTensorAccessorR make_bias(Allocator &allocator) {
  return create_1d_accessor_r_with_contents<float>({0.5, -1, 2}, allocator);
}

static GenericTensorAccessorR make_output_grad(Allocator &allocator) {
  return create_4d_accessor_r_with_contents<float>(
      {
          {
              {
                  {-0.9375, -0.875, -0.8125, -0.75},
                  {-0.6875, -0.625, -0.5625, -0.5},
                  {-0.4375, -0.375, -0.3125, -0.25},
                  {-0.1875, -0.125, -0.0625, 0},
              },
              {
                  {0.0625, 0.125, 0.1875, 0.25},
                  {0.3125, 0.375, 0.4375, 0.5},
                  {0.5625, 0.625, 0.6875, 0.75},
                  {0.8125, 0.875, 0.9375, 1},
              },
              {
                  {1.0625, 1.125, 1.1875, 1.25},
                  {1.3125, 1.375, 1.4375, 1.5},
                  {1.5625, 1.625, 1.6875, 1.75},
                  {1.8125, 1.875, 1.9375, 2},
              },
          },
      },
      allocator);
}

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("conv_2d_gpu_forward_kernel") {
    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    Conv2DAttrs attrs = make_attrs();

    GenericTensorAccessorR input = make_input(allocator);
    GenericTensorAccessorR filter = make_filter(allocator);
    GenericTensorAccessorR bias = make_bias(allocator);

    TensorShape output_shape = TensorShape{
        TensorDims{FFOrdered{1_p, 3_p, 4_p, 4_p}},
        DataType::FLOAT,
    };

    // Intentionally randomize this tensor so we can be confident we never read
    // it
    GenericTensorAccessorW output =
        create_random_filled_accessor_w(output_shape, allocator);

    Conv2DPerDeviceState per_device_state = conv_2d_gpu_init_kernel(
        managed_handle.raw_handle(), attrs, input.shape, output.shape);

    conv_2d_gpu_forward_kernel(
        /*stream=*/managed_stream.raw_stream(),
        /*handle=*/managed_handle.raw_handle(),
        /*per_device_state=*/per_device_state,
        /*attrs=*/attrs,
        /*input=*/input,
        /*filter=*/filter,
        /*bias=*/bias,
        /*output=*/output);

    GenericTensorAccessorR correct = create_4d_accessor_r_with_contents<float>(
        {
            {
                {
                    {4.5, 2.6875, 1.6875, 11.125},
                    {-2.0625, 5.9375, -4.1875, 0.875},
                    {-5, 2.4375, 3.5625, -0.4375},
                    {3.875, -5.25, 8.5, -3.0625},
                },
                {
                    {-1.5625, -2.5625, -4.0625, 1.8125},
                    {-1.28125, 4.5625, 6.40625, 2.9375},
                    {2.75, 3.625, -1.875, -4.9375},
                    {-6.0625, 1.625, 6.375, 4.25},
                },
                {
                    {2.875, 6.6875, 10.5, 6.75},
                    {9.0625, 4.8125, 2, 10.75},
                    {-3.53125, 2.09375, 3.71875, -0.5},
                    {1.5, 7.25, -2.375, 2.4375},
                },
            },
        },
        allocator);

    CHECK_MESSAGE(accessors_are_equal(output, correct),
                  check_kv("output", format_accessor_w_contents(output)));

    conv_2d_gpu_cleanup_kernel(per_device_state);
  }

  TEST_CASE("conv_2d_gpu_backward_kernel") {
    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    Conv2DAttrs attrs = make_attrs();

    GenericTensorAccessorR input = make_input(allocator);
    GenericTensorAccessorR filter = make_filter(allocator);
    GenericTensorAccessorR output_grad = make_output_grad(allocator);

    GenericTensorAccessorR output = create_4d_accessor_r_with_contents<float>(
        {
            {
                {
                    {4.5, 2.6875, 1.6875, 11.125},
                    {-2.0625, 5.9375, -4.1875, 0.875},
                    {-5, 2.4375, 3.5625, -0.4375},
                    {3.875, -5.25, 8.5, -3.0625},
                },
                {
                    {-1.5625, -2.5625, -4.0625, 1.8125},
                    {-1.28125, 4.5625, 6.40625, 2.9375},
                    {2.75, 3.625, -1.875, -4.9375},
                    {-6.0625, 1.625, 6.375, 4.25},
                },
                {
                    {2.875, 6.6875, 10.5, 6.75},
                    {9.0625, 4.8125, 2, 10.75},
                    {-3.53125, 2.09375, 3.71875, -0.5},
                    {1.5, 7.25, -2.375, 2.4375},
                },
            },
        },
        allocator);

    // The gradients are accumulated into, so they need to start from known
    // values
    GenericTensorAccessorW input_grad =
        create_4d_accessor_w_with_contents<float>(
            {
                {
                    {
                        {-2.75, -2.5, -2.25, -2},
                        {-1.75, -1.5, -1.25, -1},
                        {-0.75, -0.5, -0.25, 0},
                        {0.25, 0.5, 0.75, 1},
                    },
                    {
                        {1.25, 1.5, 1.75, 2},
                        {2.25, 2.5, 2.75, 3},
                        {3.25, 3.5, 3.75, 4},
                        {4.25, 4.5, 4.75, 5},
                    },
                },
            },
            allocator);

    GenericTensorAccessorW filter_grad = create_4d_accessor_w_with_contents<
        float>(
        {
            {
                {{-1.875, -1.75, -1.625},
                 {-1.5, -1.375, -1.25},
                 {-1.125, -1, -0.875}},
                {{-0.75, -0.625, -0.5},
                 {-0.375, -0.25, -0.125},
                 {0, 0.125, 0.25}},
            },
            {
                {{0.375, 0.5, 0.625}, {0.75, 0.875, 1}, {1.125, 1.25, 1.375}},
                {{1.5, 1.625, 1.75}, {1.875, 2, 2.125}, {2.25, 2.375, 2.5}},
            },
            {
                {{2.625, 2.75, 2.875}, {3, 3.125, 3.25}, {3.375, 3.5, 3.625}},
                {{3.75, 3.875, 4}, {4.125, 4.25, 4.375}, {4.5, 4.625, 4.75}},
            },
        },
        allocator);

    GenericTensorAccessorW bias_grad =
        create_1d_accessor_w_with_contents<float>({10, -5, 0.25}, allocator);

    Conv2DPerDeviceState per_device_state = conv_2d_gpu_init_kernel(
        managed_handle.raw_handle(), attrs, input.shape, output.shape);

    conv_2d_gpu_backward_kernel(
        /*stream=*/managed_stream.raw_stream(),
        /*handle=*/managed_handle.raw_handle(),
        /*per_device_state=*/per_device_state,
        /*attrs=*/attrs,
        /*output=*/output,
        /*output_grad=*/output_grad,
        /*input=*/input,
        /*input_grad=*/input_grad,
        /*filter=*/filter,
        /*filter_grad=*/filter_grad,
        /*bias_grad=*/bias_grad);

    GenericTensorAccessorR correct_input_grad =
        create_4d_accessor_r_with_contents<float>(
            {
                {
                    {
                        {-2.546875, -0.6875, -0.078125, 0.765625},
                        {-1.296875, -1.59375, -1.078125, -0.09375},
                        {0.640625, 0.46875, 0.984375, 2.09375},
                        {-0.640625, -2.375, -2.046875, -0.84375},
                    },
                    {
                        {4.890625, 4.359375, 4.890625, 2.03125},
                        {8.09375, 8.546875, 9.1875, 3.984375},
                        {10.34375, 11.109375, 11.75, 5.171875},
                        {8.21875, 8.015625, 8.375, 6.15625},
                    },
                },
            },
            allocator);

    GenericTensorAccessorR correct_filter_grad =
        create_4d_accessor_r_with_contents<float>(
            {
                {
                    {{-4.7734375, -6.453125, -6.5703125},
                     {-6.0859375, -8.78125, -8.5390625},
                     {-1.7421875, -0.8125, -0.5703125}},
                    {{-2.78125, -2.109375, -1.6875},
                     {-4, -3.5625, -2.5625},
                     {-3.46875, -5.4375, -2.59375}},
                },
                {
                    {{2.3515625, 3.671875, 5.0546875},
                     {5.2890625, 5.59375, 3.3359375},
                     {3.6328125, 3.5625, 2.3046875}},
                    {{4.71875, 6.390625, 5.0625},
                     {3.25, 5.6875, 4.1875},
                     {2.28125, 4.3125, 3.65625}},
                },
                {
                    {{9.4765625, 13.796875, 16.6796875},
                     {16.6640625, 19.96875, 15.2109375},
                     {9.0078125, 7.9375, 5.1796875}},
                    {{12.21875, 14.890625, 11.8125},
                     {10.5, 14.9375, 10.9375},
                     {8.03125, 14.0625, 9.90625}},
                },
            },
            allocator);

    GenericTensorAccessorR correct_bias_grad =
        create_1d_accessor_r_with_contents<float>({2.5, 3.5, 24.75}, allocator);

    CHECK_MESSAGE(
        accessors_are_equal(input_grad, correct_input_grad),
        check_kv("input_grad", format_accessor_w_contents(input_grad)));

    CHECK_MESSAGE(
        accessors_are_equal(filter_grad, correct_filter_grad),
        check_kv("filter_grad", format_accessor_w_contents(filter_grad)));

    CHECK_MESSAGE(accessors_are_equal(bias_grad, correct_bias_grad),
                  check_kv("bias_grad", format_accessor_w_contents(bias_grad)));

    conv_2d_gpu_cleanup_kernel(per_device_state);
  }
}
