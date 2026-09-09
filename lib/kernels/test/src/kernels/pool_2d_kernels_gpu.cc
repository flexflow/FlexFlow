#include "kernels/pool_2d_kernels_gpu.h"
#include "internal/test_utils.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/format_accessor_contents.h"
#include "test/utils/doctest/check_kv.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

// NCHW
static GenericTensorAccessorR make_input(Allocator &allocator) {
  return create_4d_accessor_r_with_contents<float>(
      {
          {
              {
                  {1, 2, 3, 4},
                  {5, 6, 7, 8},
                  {-1, -2, -3, -4},
                  {0.5, 0.25, 0.125, 0.0625},
              },
              {
                  {9, 1, 0.5, -7},
                  {2, 3, 0.25, 0.125},
                  {-8, 4, 10, 20},
                  {6, -5, 30, 40},
              },
          },
      },
      allocator);
}

// input_grad is accumulated into, so it needs to start from a known value
static GenericTensorAccessorW make_input_grad(Allocator &allocator) {
  return create_4d_accessor_w_with_contents<float>(
      {
          {
              {
                  {-4.5, -4, -3.5, -3},
                  {-2.5, -2, -1.5, -1},
                  {-0.5, 0, 0.5, 1},
                  {1.5, 2, 2.5, 3},
              },
              {
                  {3.5, 4, 4.5, 5},
                  {5.5, 6, 6.5, 7},
                  {7.5, 8, 8.5, 9},
                  {9.5, 10, 10.5, 11},
              },
          },
      },
      allocator);
}

static GenericTensorAccessorR make_output_grad(Allocator &allocator) {
  return create_4d_accessor_r_with_contents<float>(
      {
          {
              {
                  {-0.75, -0.5},
                  {-0.25, 0},
              },
              {
                  {0.25, 0.5},
                  {0.75, 1},
              },
          },
      },
      allocator);
}

static Pool2DAttrs make_attrs(PoolOp pool_type) {
  return Pool2DAttrs{
      /*kernel_h=*/2_p,
      /*kernel_w=*/2_p,
      /*stride_h=*/2_p,
      /*stride_w=*/2_p,
      /*padding_h=*/0_n,
      /*padding_w=*/0_n,
      /*pool_type=*/pool_type,
      /*activation=*/std::nullopt,
  };
}

static TensorShape output_shape() {
  return TensorShape{
      TensorDims{FFOrdered{1_p, 2_p, 2_p, 2_p}},
      DataType::FLOAT,
  };
}

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("pool_2d_gpu_forward_kernel") {
    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    GenericTensorAccessorR input = make_input(allocator);

    SUBCASE("PoolOp::MAX") {
      Pool2DAttrs attrs = make_attrs(PoolOp::MAX);

      // Intentionally randomize this tensor so we can be confident we never
      // read it
      GenericTensorAccessorW output =
          create_random_filled_accessor_w(output_shape(), allocator);

      Pool2DPerDeviceState per_device_state =
          pool_2d_gpu_init_kernel(attrs, input.shape, output.shape);

      pool_2d_gpu_forward_kernel(
          /*stream=*/managed_stream.raw_stream(),
          /*handle=*/managed_handle.raw_handle(),
          /*per_device_state=*/per_device_state,
          /*attrs=*/attrs,
          /*input=*/input,
          /*output=*/output);

      GenericTensorAccessorR correct =
          create_4d_accessor_r_with_contents<float>(
              {
                  {
                      {{6, 8}, {0.5, 0.125}},
                      {{9, 0.5}, {6, 40}},
                  },
              },
              allocator);

      CHECK_MESSAGE(accessors_are_equal(output, correct),
                    check_kv("output", format_accessor_w_contents(output)));
    }

    SUBCASE("PoolOp::AVG") {
      Pool2DAttrs attrs = make_attrs(PoolOp::AVG);

      // Intentionally randomize this tensor so we can be confident we never
      // read it
      GenericTensorAccessorW output =
          create_random_filled_accessor_w(output_shape(), allocator);

      Pool2DPerDeviceState per_device_state =
          pool_2d_gpu_init_kernel(attrs, input.shape, output.shape);

      pool_2d_gpu_forward_kernel(
          /*stream=*/managed_stream.raw_stream(),
          /*handle=*/managed_handle.raw_handle(),
          /*per_device_state=*/per_device_state,
          /*attrs=*/attrs,
          /*input=*/input,
          /*output=*/output);

      GenericTensorAccessorR correct =
          create_4d_accessor_r_with_contents<float>(
              {
                  {
                      {{3.5, 5.5}, {-0.5625, -1.703125}},
                      {{3.75, -1.53125}, {-0.75, 25}},
                  },
              },
              allocator);

      CHECK_MESSAGE(accessors_are_equal(output, correct),
                    check_kv("output", format_accessor_w_contents(output)));
    }
  }

  TEST_CASE("pool_2d_gpu_backward_kernel") {
    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    GenericTensorAccessorR input = make_input(allocator);
    GenericTensorAccessorR output_grad = make_output_grad(allocator);

    SUBCASE("PoolOp::MAX") {
      Pool2DAttrs attrs = make_attrs(PoolOp::MAX);

      GenericTensorAccessorR output = create_4d_accessor_r_with_contents<float>(
          {
              {
                  {{6, 8}, {0.5, 0.125}},
                  {{9, 0.5}, {6, 40}},
              },
          },
          allocator);

      GenericTensorAccessorW input_grad = make_input_grad(allocator);

      Pool2DPerDeviceState per_device_state =
          pool_2d_gpu_init_kernel(attrs, input.shape, output.shape);

      pool_2d_gpu_backward_kernel(
          /*stream=*/managed_stream.raw_stream(),
          /*handle=*/managed_handle.raw_handle(),
          /*per_device_state=*/per_device_state,
          /*attrs=*/attrs,
          /*output=*/output,
          /*output_grad=*/output_grad,
          /*input=*/input,
          /*input_grad=*/input_grad);

      GenericTensorAccessorR correct =
          create_4d_accessor_r_with_contents<float>(
              {
                  {
                      {
                          {-4.5, -4, -3.5, -3},
                          {-2.5, -2.75, -1.5, -1.5},
                          {-0.5, 0, 0.5, 1},
                          {1.25, 2, 2.5, 3},
                      },
                      {
                          {3.75, 4, 5, 5},
                          {5.5, 6, 6.5, 7},
                          {7.5, 8, 8.5, 9},
                          {10.25, 10, 10.5, 12},
                      },
                  },
              },
              allocator);

      CHECK_MESSAGE(
          accessors_are_equal(input_grad, correct),
          check_kv("input_grad", format_accessor_w_contents(input_grad)));
    }

    SUBCASE("PoolOp::AVG") {
      Pool2DAttrs attrs = make_attrs(PoolOp::AVG);

      GenericTensorAccessorR output = create_4d_accessor_r_with_contents<float>(
          {
              {
                  {{3.5, 5.5}, {-0.5625, -1.703125}},
                  {{3.75, -1.53125}, {-0.75, 25}},
              },
          },
          allocator);

      GenericTensorAccessorW input_grad = make_input_grad(allocator);

      Pool2DPerDeviceState per_device_state =
          pool_2d_gpu_init_kernel(attrs, input.shape, output.shape);

      pool_2d_gpu_backward_kernel(
          /*stream=*/managed_stream.raw_stream(),
          /*handle=*/managed_handle.raw_handle(),
          /*per_device_state=*/per_device_state,
          /*attrs=*/attrs,
          /*output=*/output,
          /*output_grad=*/output_grad,
          /*input=*/input,
          /*input_grad=*/input_grad);

      GenericTensorAccessorR correct =
          create_4d_accessor_r_with_contents<float>(
              {
                  {
                      {
                          {-4.6875, -4.1875, -3.625, -3.125},
                          {-2.6875, -2.1875, -1.625, -1.125},
                          {-0.5625, -0.0625, 0.5, 1},
                          {1.4375, 1.9375, 2.5, 3},
                      },
                      {
                          {3.5625, 4.0625, 4.625, 5.125},
                          {5.5625, 6.0625, 6.625, 7.125},
                          {7.6875, 8.1875, 8.75, 9.25},
                          {9.6875, 10.1875, 10.75, 11.25},
                      },
                  },
              },
              allocator);

      CHECK_MESSAGE(
          accessors_are_equal(input_grad, correct),
          check_kv("input_grad", format_accessor_w_contents(input_grad)));
    }
  }
}
