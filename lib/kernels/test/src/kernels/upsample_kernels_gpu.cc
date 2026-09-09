#include "kernels/upsample_kernels_gpu.h"
#include "internal/test_utils.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/format_accessor_contents.h"
#include "test/utils/doctest/check_kv.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

static UpsampleAttrs make_attrs(int scale_factor) {
  return UpsampleAttrs{
      /*scale_factor=*/int_ge_two{scale_factor},
      /*mode=*/UpsampleMode::NEAREST,
  };
}

// NCHW
static GenericTensorAccessorR make_input(Allocator &allocator) {
  return create_4d_accessor_r_with_contents<float>(
      {
          {
              {
                  {1, 2, 3},
                  {4, 5, 6},
              },
              {
                  {-1, -2, -3},
                  {0.5, 0.25, 0.125},
              },
          },
      },
      allocator);
}

static GenericTensorAccessorR make_output(Allocator &allocator) {
  return create_4d_accessor_r_with_contents<float>(
      {
          {
              {
                  {1, 1, 2, 2, 3, 3},
                  {1, 1, 2, 2, 3, 3},
                  {4, 4, 5, 5, 6, 6},
                  {4, 4, 5, 5, 6, 6},
              },
              {
                  {-1, -1, -2, -2, -3, -3},
                  {-1, -1, -2, -2, -3, -3},
                  {0.5, 0.5, 0.25, 0.25, 0.125, 0.125},
                  {0.5, 0.5, 0.25, 0.25, 0.125, 0.125},
              },
          },
      },
      allocator);
}

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("upsample_gpu_forward_kernel") {
    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    UpsampleAttrs attrs = make_attrs(/*scale_factor=*/2);

    GenericTensorAccessorR input = make_input(allocator);

    TensorShape output_shape = TensorShape{
        TensorDims{FFOrdered{1_p, 2_p, 4_p, 6_p}},
        DataType::FLOAT,
    };

    // Intentionally randomize this tensor so we can be confident we never read
    // it
    GenericTensorAccessorW output =
        create_random_filled_accessor_w(output_shape, allocator);

    upsample_gpu_forward_kernel(
        /*stream=*/managed_stream.raw_stream(),
        /*attrs=*/attrs,
        /*input=*/input,
        /*output=*/output);

    GenericTensorAccessorR correct = make_output(allocator);

    CHECK_MESSAGE(accessors_are_equal(output, correct),
                  check_kv("output", format_accessor_w_contents(output)));
  }

  TEST_CASE("upsample_gpu_backward_kernel") {
    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    UpsampleAttrs attrs = make_attrs(/*scale_factor=*/2);

    GenericTensorAccessorR input = make_input(allocator);
    GenericTensorAccessorR output = make_output(allocator);

    GenericTensorAccessorR output_grad =
        create_4d_accessor_r_with_contents<float>(
            {
                {
                    {
                        {-0.875, -0.75, -0.625, -0.5, -0.375, -0.25},
                        {-0.125, 0, 0.125, 0.25, 0.375, 0.5},
                        {0.625, 0.75, 0.875, 1, 1.125, 1.25},
                        {1.375, 1.5, 1.625, 1.75, 1.875, 2},
                    },
                    {
                        {2.125, 2.25, 2.375, 2.5, 2.625, 2.75},
                        {2.875, 3, 3.125, 3.25, 3.375, 3.5},
                        {3.625, 3.75, 3.875, 4, 4.125, 4.25},
                        {4.375, 4.5, 4.625, 4.75, 4.875, 5},
                    },
                },
            },
            allocator);

    // input_grad is accumulated into, so it needs to start from a known value
    GenericTensorAccessorW input_grad =
        create_4d_accessor_w_with_contents<float>(
            {
                {
                    {
                        {-7, -4, -1},
                        {2, 5, 8},
                    },
                    {
                        {11, 14, 17},
                        {20, 23, 26},
                    },
                },
            },
            allocator);

    upsample_gpu_backward_kernel(
        /*stream=*/managed_stream.raw_stream(),
        /*attrs=*/attrs,
        /*output=*/output,
        /*output_grad=*/output_grad,
        /*input=*/input,
        /*input_grad=*/input_grad);

    GenericTensorAccessorR correct = create_4d_accessor_r_with_contents<float>(
        {
            {
                {
                    {-8.75, -4.75, -0.75},
                    {6.25, 10.25, 14.25},
                },
                {
                    {21.25, 25.25, 29.25},
                    {36.25, 40.25, 44.25},
                },
            },
        },
        allocator);

    CHECK_MESSAGE(
        accessors_are_equal(input_grad, correct),
        check_kv("input_grad", format_accessor_w_contents(input_grad)));
  }

  TEST_CASE("upsample_gpu_forward_kernel (scale_factor = 3)") {
    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    UpsampleAttrs attrs = make_attrs(/*scale_factor=*/3);

    GenericTensorAccessorR input = create_4d_accessor_r_with_contents<float>(
        {
            {
                {
                    {1, 2},
                    {-3, 0.5},
                },
            },
        },
        allocator);

    TensorShape output_shape = TensorShape{
        TensorDims{FFOrdered{1_p, 1_p, 6_p, 6_p}},
        DataType::FLOAT,
    };

    // Intentionally randomize this tensor so we can be confident we never read
    // it
    GenericTensorAccessorW output =
        create_random_filled_accessor_w(output_shape, allocator);

    upsample_gpu_forward_kernel(
        /*stream=*/managed_stream.raw_stream(),
        /*attrs=*/attrs,
        /*input=*/input,
        /*output=*/output);

    GenericTensorAccessorR correct = create_4d_accessor_r_with_contents<float>(
        {
            {
                {
                    {1, 1, 1, 2, 2, 2},
                    {1, 1, 1, 2, 2, 2},
                    {1, 1, 1, 2, 2, 2},
                    {-3, -3, -3, 0.5, 0.5, 0.5},
                    {-3, -3, -3, 0.5, 0.5, 0.5},
                    {-3, -3, -3, 0.5, 0.5, 0.5},
                },
            },
        },
        allocator);

    CHECK_MESSAGE(accessors_are_equal(output, correct),
                  check_kv("output", format_accessor_w_contents(output)));
  }

  TEST_CASE("upsample_gpu_backward_kernel (scale_factor = 3)") {
    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    UpsampleAttrs attrs = make_attrs(/*scale_factor=*/3);

    GenericTensorAccessorR input = create_4d_accessor_r_with_contents<float>(
        {
            {
                {
                    {1, 2},
                    {-3, 0.5},
                },
            },
        },
        allocator);

    GenericTensorAccessorR output = create_4d_accessor_r_with_contents<float>(
        {
            {
                {
                    {1, 1, 1, 2, 2, 2},
                    {1, 1, 1, 2, 2, 2},
                    {1, 1, 1, 2, 2, 2},
                    {-3, -3, -3, 0.5, 0.5, 0.5},
                    {-3, -3, -3, 0.5, 0.5, 0.5},
                    {-3, -3, -3, 0.5, 0.5, 0.5},
                },
            },
        },
        allocator);

    GenericTensorAccessorR output_grad =
        create_4d_accessor_r_with_contents<float>(
            {
                {
                    {
                        {-3.75, -3.5, -3.25, -3, -2.75, -2.5},
                        {-2.25, -2, -1.75, -1.5, -1.25, -1},
                        {-0.75, -0.5, -0.25, 0, 0.25, 0.5},
                        {0.75, 1, 1.25, 1.5, 1.75, 2},
                        {2.25, 2.5, 2.75, 3, 3.25, 3.5},
                        {3.75, 4, 4.25, 4.5, 4.75, 5},
                    },
                },
            },
            allocator);

    // input_grad is accumulated into, so it needs to start from a known value
    GenericTensorAccessorW input_grad =
        create_4d_accessor_w_with_contents<float>(
            {
                {
                    {
                        {100, 200},
                        {300, 400},
                    },
                },
            },
            allocator);

    upsample_gpu_backward_kernel(
        /*stream=*/managed_stream.raw_stream(),
        /*attrs=*/attrs,
        /*output=*/output,
        /*output_grad=*/output_grad,
        /*input=*/input,
        /*input_grad=*/input_grad);

    GenericTensorAccessorR correct = create_4d_accessor_r_with_contents<float>(
        {
            {
                {
                    {82, 188.75},
                    {322.5, 429.25},
                },
            },
        },
        allocator);

    CHECK_MESSAGE(
        accessors_are_equal(input_grad, correct),
        check_kv("input_grad", format_accessor_w_contents(input_grad)));
  }
}
