#include "kernels/linear_kernels.h"
#include "kernels/accessors_are_equal.h"
#include "kernels/copy_tensor_accessor.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/create_local_allocator_for_device_type.h"
#include "kernels/create_zero_filled_accessor.h"
#include "kernels/device_handle_t.h"
#include "kernels/device_stream_t.h"
#include "kernels/format_accessor_contents.h"
#include "kernels/managed_per_device_ff_handle.h"
#include "test/utils/doctest/check_kv.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("linear_forward_kernel cpu-gpu alignment") {
    Allocator local_cpu_allocator = create_local_cpu_memory_allocator();

    // GenericTensorAccessorR toy_input =
    // create_2d_accessor_r_with_contents<float>(
    //     {
    //         {3, 3, 6},
    //         {2, 1, 5},
    //         {1, 2, -2},
    //         {8, 0.5, -3},
    //     },
    //     local_cpu_allocator);
    // float const *toy_arr = toy_input.get_float_ptr();
    // std::cout << toy_arr[0] << "  "
    //           << toy_arr[1] << "  "
    //           << toy_arr[2] << std::endl;
    //
    // Allocator local_cuda_allocator = create_local_cuda_memory_allocator();
    // GenericTensorAccessorW toy_cuda =
    // local_cuda_allocator.allocate_tensor(toy_input.shape);
    // copy_accessor_data_to_l_from_r(toy_cuda, toy_input);
    // GenericTensorAccessorW toy_input2 =
    // local_cpu_allocator.allocate_tensor(toy_input.shape);
    // copy_accessor_data_to_l_from_r(toy_input2,
    // read_only_accessor_from_write_accessor(toy_cuda)); CHECK_MESSAGE(
    //     accessors_are_equal(toy_input, toy_input2),
    //     check_kv("cpu_result", format_accessor_r_contents(toy_input)),
    //     check_kv("gpu_result", format_accessor_w_contents(toy_input2)));

    auto run_forward_kernel = [&](DeviceType device_type) {
      Allocator allocator = create_local_allocator_for_device_type(device_type);

      GenericTensorAccessorR input = create_2d_accessor_r_with_contents<float>(
          {
              {3, 3, 6},
              {2, 1, 5},
              {1, 2, -2},
              {8, 0.5, -3},
          },
          allocator);

      GenericTensorAccessorR projection =
          create_2d_accessor_r_with_contents<float>(
              {
                  {1.0f, 2.0f, 1.5f},
                  {0.5f, 4.0f, -1.0f},
              },
              allocator);

      GenericTensorAccessorR bias =
          create_1d_accessor_r_with_contents<float>({3.0, -1.0}, allocator);

      int batch_size = 4;
      positive_int output_num_channels = 2_p;

      TensorShape output_shape = TensorShape{
          TensorDims{FFOrdered{positive_int{batch_size}, output_num_channels}},
          DataType::FLOAT,
      };

      GenericTensorAccessorW output = allocator.allocate_tensor(output_shape);

      std::optional<ManagedPerDeviceFFHandle> managed_handle =
          create_local_handle_for_device_type(
              device_type,
              /*workSpaceSize=*/1024 * 1024,
              /*allowTensorOpMathConversion=*/true);

      LinearAttrs attrs = LinearAttrs{
          /*out_channels=*/output_num_channels,
          /*use_bias=*/true,
          /*data_type=*/DataType::FLOAT,
          /*activation=*/std::nullopt,
          /*regularizer=*/std::nullopt,
      };

      std::optional<LinearPerDeviceState> per_device_state = linear_init_kernel(
          /*device_type=*/device_type,
          /*handle=*/device_handle_t_from_managed_ff_handle(managed_handle),
          /*activation=*/attrs.activation,
          /*regularizer=*/attrs.regularizer,
          /*use_bias=*/attrs.use_bias,
          /*input_type=*/DataType::FLOAT,
          /*weight_type=*/DataType::FLOAT,
          /*output_type=*/DataType::FLOAT,
          /*batch_size=*/batch_size,
          /*output_num_channels=*/attrs.out_channels.int_from_positive_int());

      device_stream_t stream = get_stream_for_device_type(device_type);

      linear_forward_kernel(
          /*stream=*/stream,
          /*per_device_state=*/per_device_state,
          /*attrs=*/attrs,
          /*input_accessor=*/input,
          /*output_accessor=*/output,
          /*projection_accessor=*/projection,
          /*bias_accessor=*/bias);

      return copy_tensor_accessor_w(output, local_cpu_allocator);
    };

    GenericTensorAccessorW cpu_result = run_forward_kernel(DeviceType::CPU);
    GenericTensorAccessorW gpu_result = run_forward_kernel(DeviceType::GPU);

    CHECK_MESSAGE(
        accessors_are_equal(cpu_result, gpu_result),
        check_kv("cpu_result", format_accessor_w_contents(cpu_result)),
        check_kv("gpu_result", format_accessor_w_contents(gpu_result)));
  }

  TEST_CASE("backward_kernel CPU/GPU alignment (Linear)") {
    Allocator local_cpu_allocator = create_local_cpu_memory_allocator();

    auto run_forward_kernel = [&](DeviceType device_type) {
      Allocator allocator = create_local_allocator_for_device_type(device_type);

      GenericTensorAccessorR input = create_2d_accessor_r_with_contents<float>(
          {
              {3, 3, 6},
              {2, 1, 5},
              {1, 2, -2},
              {8, 0.5, -3},
          },
          allocator);

      GenericTensorAccessorW input_grad = create_zero_filled_accessor_w(
          get_tensor_shape_for_accessor_r(input), allocator);

      GenericTensorAccessorR projection =
          create_2d_accessor_r_with_contents<float>(
              {
                  {1.0f, 2.0f, 1.5f},
                  {0.5f, 4.0f, -1.0f},
              },
              allocator);

      GenericTensorAccessorW projection_grad = create_zero_filled_accessor_w(
          get_tensor_shape_for_accessor_r(projection), allocator);

      GenericTensorAccessorR bias =
          create_1d_accessor_r_with_contents<float>({3.0, -1.0}, allocator);

      GenericTensorAccessorW bias_grad = create_zero_filled_accessor_w(
          get_tensor_shape_for_accessor_r(bias), allocator);

      GenericTensorAccessorR output = create_2d_accessor_r_with_contents<float>(
          {
              {21.0f, 6.5f},
              {14.5f, -1.0f},
              {5.0f, 9.5f},
              {7.5f, 8.0f},
          },
          allocator);

      GenericTensorAccessorR output_grad =
          create_2d_accessor_r_with_contents<float>(
              {
                  {1.0f, -0.5f},
                  {2.0f, -2.0f},
                  {1.0f, 9.0f},
                  {-3.5f, 1.0f},
              },
              allocator);

      int batch_size = 4;
      positive_int output_num_channels = 2_p;

      LinearAttrs attrs = LinearAttrs{
          /*out_channels=*/output_num_channels,
          /*use_bias=*/true,
          /*data_type=*/DataType::FLOAT,
          /*activation=*/std::nullopt,
          /*regularizer=*/std::nullopt,
      };

      TensorShape output_shape = TensorShape{
          TensorDims{FFOrdered{positive_int{batch_size},
                               positive_int{output_num_channels}}},
          DataType::FLOAT,
      };

      std::optional<ManagedPerDeviceFFHandle> managed_handle =
          create_local_handle_for_device_type(
              device_type,
              /*workSpaceSize=*/1024 * 1024,
              /*allowTensorOpMathConversion=*/true);

      std::optional<LinearPerDeviceState> per_device_state = linear_init_kernel(
          /*device_type=*/device_type,
          /*handle=*/device_handle_t_from_managed_ff_handle(managed_handle),
          /*activation=*/attrs.activation,
          /*regularizer=*/attrs.regularizer,
          /*use_bias=*/true,
          /*input_type=*/DataType::FLOAT,
          /*weight_type=*/DataType::FLOAT,
          /*output_type=*/DataType::FLOAT,
          /*batch_size=*/batch_size,
          /*output_num_channels=*/attrs.out_channels.int_from_positive_int());

      device_stream_t stream = get_stream_for_device_type(device_type);

      linear_backward_kernel(
          /*stream=*/stream,
          /*per_device_state=*/per_device_state,
          /*attrs=*/attrs,
          /*output=*/output,
          /*output_grad=*/output_grad,
          /*input=*/input,
          /*input_grad=*/input_grad,
          /*projection=*/projection,
          /*projection_grad=*/projection_grad,
          /*bias_grad=*/bias_grad);

      return std::tuple{
          copy_tensor_accessor_w(input_grad, local_cpu_allocator),
          copy_tensor_accessor_w(projection_grad, local_cpu_allocator),
          copy_tensor_accessor_w(bias_grad, local_cpu_allocator),
      };
    };

    auto cpu_results = run_forward_kernel(DeviceType::CPU);
    GenericTensorAccessorW cpu_input_grad = std::get<0>(cpu_results);
    GenericTensorAccessorW cpu_projection_grad = std::get<1>(cpu_results);
    GenericTensorAccessorW cpu_bias_grad = std::get<2>(cpu_results);

    auto gpu_results = run_forward_kernel(DeviceType::GPU);
    GenericTensorAccessorW gpu_input_grad = std::get<0>(gpu_results);
    GenericTensorAccessorW gpu_projection_grad = std::get<1>(gpu_results);
    GenericTensorAccessorW gpu_bias_grad = std::get<2>(gpu_results);

    CHECK_MESSAGE(
        accessors_are_equal(cpu_input_grad, gpu_input_grad),
        check_kv("cpu_input_grad", format_accessor_w_contents(cpu_input_grad)),
        check_kv("gpu_input_grad", format_accessor_w_contents(gpu_input_grad)));

    CHECK_MESSAGE(accessors_are_equal(cpu_projection_grad, gpu_projection_grad),
                  check_kv("cpu_projection_grad",
                           format_accessor_w_contents(cpu_projection_grad)),
                  check_kv("gpu_projection_grad",
                           format_accessor_w_contents(gpu_projection_grad)));

    CHECK_MESSAGE(
        accessors_are_equal(cpu_bias_grad, gpu_bias_grad),
        check_kv("cpu_bias_grad", format_accessor_w_contents(cpu_bias_grad)),
        check_kv("gpu_bias_grad", format_accessor_w_contents(gpu_bias_grad)));
  }
}
