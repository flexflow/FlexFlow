#include "kernels/accessors_are_equal.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/create_random_filled_accessor.h"
#include "kernels/element_unary_kernels_cpu.h"
#include "kernels/format_accessor_contents.h"
#include "kernels/local_cpu_allocator.h"
#include "kernels/local_cuda_allocator.h"
#include "kernels/managed_ff_stream.h"
#include "kernels/managed_per_device_ff_handle.h"
#include "op-attrs/ops/element_unary.h"
#include "test/utils/doctest/check_kv.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("element_unary_gpu_forward_kernel") {
    SUBCASE("relu") {
      ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
          /*workSpaceSize=*/1024 * 1024,
          /*allowTensorOpMathConversion=*/true);
      ManagedFFStream managed_stream{};

      Allocator allocator = create_local_cuda_memory_allocator();

      ElementUnaryAttrs attrs = make_relu_attrs();

      GenericTensorAccessorR input = create_2d_accessor_r_with_contents<float>(
          {
              {3, -3, 6},
              {0, 1, 5},
              {1, 2, -2},
              {-8, 0.5, -3},
          },
          allocator);

      // Intentionally randomize this tensor so we can be confident we never read it
      GenericTensorAccessorW result =
          create_random_filled_accessor_w(input.shape, allocator);

      element_unary_cpu_forward_kernel(
          /*attrs=*/attrs,
          /*input=*/input,
          /*output=*/result);

      GenericTensorAccessorR correct =
          create_2d_accessor_r_with_contents<float>(
              {
                  {3, 0, 6},
                  {0, 1, 5},
                  {1, 2, 0},
                  {0, 0.5, 0},
              },
              allocator);

      CHECK_MESSAGE(accessors_are_equal(result, correct),
                    check_kv("result", format_accessor_w_contents(result)));
    }
  }

  TEST_CASE("element_unary_gpu_backward_kernel") {
    SUBCASE("relu") {
      ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
          /*workSpaceSize=*/1024 * 1024,
          /*allowTensorOpMathConversion=*/true);
      ManagedFFStream managed_stream{};

      Allocator allocator = create_local_cuda_memory_allocator();

      ElementUnaryAttrs attrs = make_relu_attrs();

      GenericTensorAccessorR input = create_2d_accessor_r_with_contents<float>(
          {
              {3, -3, 6},
              {0, 1, 5},
              {1, 2, -2},
              {-8, 0.5, -3},
          },
          allocator);

      // Intentionally randomize this tensor so we can be confident we never read it
      GenericTensorAccessorW input_grad = create_random_filled_accessor_w(
          get_tensor_shape_for_accessor_r(input), allocator);

      GenericTensorAccessorR output = create_2d_accessor_r_with_contents<float>(
          {
              {3, 0, 6},
              {0, 1, 5},
              {1, 2, -2},
              {0, 0.5, 0},
          },
          allocator);

      GenericTensorAccessorR output_grad =
          create_2d_accessor_r_with_contents<float>(
              {
                  {1, 2, -1},
                  {6, 4, 2},
                  {0.5, 0.1, -2},
                  {0, 0.5, 0},
              },
              allocator);

      element_unary_cpu_backward_kernel(
          /*attrs=*/attrs,
          /*output=*/output,
          /*output_grad=*/output_grad,
          /*input=*/input,
          /*input_grad=*/input_grad);

      GenericTensorAccessorR correct_input_grad =
          create_2d_accessor_r_with_contents<float>(
              {
                  {1.0f, 0.0f, -1.0f},
                  {0.0f, 4.0f, 2.0f},
                  {0.5f, 0.1f, 0.0f},
                  {0.0f, 0.5f, 0.0f},
              },
              allocator);

      CHECK_MESSAGE(
          accessors_are_equal(input_grad, correct_input_grad),
          check_kv("input_grad", format_accessor_w_contents(input_grad)));
    }
  }
}
