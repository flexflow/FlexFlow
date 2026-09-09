#include "internal/test_utils.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/format_accessor_contents.h"
#include "kernels/softmax_kernels_gpu.h"
#include "test/utils/doctest/check_kv.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("Softmax Kernel (GPU)") {
    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    GenericTensorAccessorR input = create_2d_accessor_r_with_contents<float>(
        {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
            {-1, -2, -3},
        },
        allocator);

    SUBCASE("softmax_gpu_forward_kernel(dim = 0)") {
      SoftmaxAttrs attrs{
          /*dim=*/ff_dim_t{0_n},
      };

      // Intentionally randomize this tensor so we can be confident we never read it
      GenericTensorAccessorW output =
          create_random_filled_accessor_w(input.shape, allocator);

      SoftmaxPerDeviceState per_device_state =
          softmax_gpu_init_kernel(attrs, input.shape, output.shape);

      softmax_gpu_forward_kernel(
          /*stream=*/managed_stream.raw_stream(),
          /*handle=*/managed_handle.raw_handle(),
          /*per_device_state=*/per_device_state,
          /*attrs=*/attrs,
          /*input=*/input,
          /*output=*/output);

      GenericTensorAccessorR correct =
          create_2d_accessor_r_with_contents<float>(
              {
                  {0.0023548827, 0.0023555316, 0.0023556193},
                  {0.04729908, 0.047312114, 0.047313876},
                  {0.9500274, 0.95028925, 0.95032465},
                  {0.00031869867, 4.3143067e-05, 5.8389965e-06},
              },
              allocator);

      CHECK_MESSAGE(accessors_are_equal(output, correct),
                    check_kv("output", format_accessor_w_contents(output)));
    }

    SUBCASE("softmax_gpu_forward_kernel(dim = 1)") {
      SoftmaxAttrs attrs{
          /*dim=*/ff_dim_t{1_n},
      };

      // Intentionally randomize this tensor so we can be confident we never read it
      GenericTensorAccessorW output =
          create_random_filled_accessor_w(input.shape, allocator);

      SoftmaxPerDeviceState per_device_state =
          softmax_gpu_init_kernel(attrs, input.shape, output.shape);

      softmax_gpu_forward_kernel(
          /*stream=*/managed_stream.raw_stream(),
          /*handle=*/managed_handle.raw_handle(),
          /*per_device_state=*/per_device_state,
          /*attrs=*/attrs,
          /*input=*/input,
          /*output=*/output);

      GenericTensorAccessorR correct =
          create_2d_accessor_r_with_contents<float>(
              {
                  {0.090030566, 0.24472848, 0.6652409},
                  {0.090030566, 0.24472848, 0.6652409},
                  {0.090030566, 0.24472848, 0.6652409},
                  {0.6652409, 0.24472848, 0.090030566},
              },
              allocator);

      CHECK_MESSAGE(accessors_are_equal(output, correct),
                    check_kv("output", format_accessor_w_contents(output)));
    }

    SUBCASE("softmax_gpu_backward_kernel(dim = 0)") {
      SoftmaxAttrs attrs{
          /*dim=*/ff_dim_t{0_n},
      };

      GenericTensorAccessorR output = create_2d_accessor_r_with_contents<float>(
          {
              {0.0023548827, 0.0023555316, 0.0023556193},
              {0.04729908, 0.047312114, 0.047313876},
              {0.9500274, 0.95028925, 0.95032465},
              {0.00031869867, 4.3143067e-05, 5.8389965e-06},
          },
          allocator);

      GenericTensorAccessorR output_grad =
          create_2d_accessor_r_with_contents<float>(
              {
                  {1, 2, 3},
                  {4, 5, 6},
                  {7, 8, 9},
                  {-1, -2, -3},
              },
              allocator);
      GenericTensorAccessorW input_grad =
          create_zero_filled_accessor_w(input.shape, allocator);

      SoftmaxPerDeviceState per_device_state =
          softmax_gpu_init_kernel(attrs, input.shape, output.shape);

      softmax_gpu_backward_kernel(
          /*stream=*/managed_stream.raw_stream(),
          /*handle=*/managed_handle.raw_handle(),
          /*per_device_state=*/per_device_state,
          /*attrs=*/attrs,
          /*output=*/output,
          /*output_grad=*/output_grad,
          /*input=*/input,
          /*input_grad=*/input_grad);

      GenericTensorAccessorR correct =
          create_2d_accessor_r_with_contents<float>(
              {
                  {-0.013755869, -0.013764547, -0.013765898},
                  {-0.13439676, -0.13453196, -0.1345538},
                  {0.1506511, 0.14872104, 0.14838853},
                  {-0.0024990516, -0.00042467876, -6.915622e-05},
              },
              allocator);

      CHECK_MESSAGE(
          accessors_are_equal(input_grad, correct),
          check_kv("input_grad", format_accessor_w_contents(input_grad)));
    }
    SUBCASE("softmax_gpu_backward_kernel(dim = 1)") {
      SoftmaxAttrs attrs{
          /*dim=*/ff_dim_t{1_n},
      };

      GenericTensorAccessorR output = create_2d_accessor_r_with_contents<float>(
          {
              {0.090030566, 0.24472848, 0.6652409},
              {0.090030566, 0.24472848, 0.6652409},
              {0.090030566, 0.24472848, 0.6652409},
              {0.6652409, 0.24472848, 0.090030566},
          },
          allocator);

      GenericTensorAccessorR output_grad =
          create_2d_accessor_r_with_contents<float>(
              {
                  {1, 2, 3},
                  {4, 5, 6},
                  {7, 8, 9},
                  {-1, -2, -3},
              },
              allocator);
      GenericTensorAccessorW input_grad =
          create_zero_filled_accessor_w(input.shape, allocator);

      SoftmaxPerDeviceState per_device_state =
          softmax_gpu_init_kernel(attrs, input.shape, output.shape);

      softmax_gpu_backward_kernel(
          /*stream=*/managed_stream.raw_stream(),
          /*handle=*/managed_handle.raw_handle(),
          /*per_device_state=*/per_device_state,
          /*attrs=*/attrs,
          /*output=*/output,
          /*output_grad=*/output_grad,
          /*input=*/input,
          /*input_grad=*/input_grad);

      GenericTensorAccessorR correct =
          create_2d_accessor_r_with_contents<float>(
              {
                  {-0.14181706, -0.14077029, 0.28258762},
                  {-0.14181706, -0.14077029, 0.28258762},
                  {-0.14181702, -0.14077017, 0.28258792},
                  {0.28258738, -0.14077038, -0.1418171},
              },
              allocator);

      CHECK_MESSAGE(
          accessors_are_equal(input_grad, correct),
          check_kv("input_grad", format_accessor_w_contents(input_grad)));
    }
  }
}
