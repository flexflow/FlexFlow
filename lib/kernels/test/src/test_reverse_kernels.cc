#include "doctest/doctest.h"
#include "kernels/reverse_kernels.h"
#include "kernels/reverse_kernels_cpu.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Call Reverse Forward and Backward Kernels") {
    std::size_t reverse_dim_size = 10;
    std::size_t in_blk_size = 10;
    std::size_t num_out_blks = 1;

    TensorShape input_shape = make_tensor_shape_from_legion_dims(
        {num_out_blks, reverse_dim_size, in_blk_size}, DataType::FLOAT);
    TensorShape output_shape = input_shape;

    ManagedPerDeviceFFHandle managed_handle{};
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    SUBCASE("forward_kernel") {
      GenericTensorAccessorR input_accessor =
          read_only_accessor_from_write_accessor(
              create_filled_accessor_w<float>(input_shape, allocator, 1.0f));
      GenericTensorAccessorW output_accessor =
          allocator.allocate_tensor(output_shape);

      Kernels::Reverse::forward_kernel(managed_stream.raw_stream(),
                                       input_accessor.get_float_ptr(),
                                       output_accessor.get_float_ptr(),
                                       num_out_blks,
                                       reverse_dim_size,
                                       in_blk_size,
                                       input_accessor.shape.num_elements());

      std::vector<float> check_output_data =
          load_accessor_data<DataType::FLOAT>(output_accessor);

      CHECK(contains_non_zero(check_output_data));
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorW output_grad_accessor =
          create_random_filled_accessor_w<DataType::FLOAT>(output_shape,
                                                           allocator);
      GenericTensorAccessorW input_grad_accessor =
          allocator.allocate_tensor(input_shape);

      Kernels::Reverse::backward_kernel(
          managed_stream.raw_stream(),
          output_grad_accessor.get_float_ptr(),
          input_grad_accessor.get_float_ptr(),
          num_out_blks,
          reverse_dim_size,
          in_blk_size,
          input_grad_accessor.shape.num_elements());

      std::vector<float> host_grad_input_data =
          load_accessor_data<DataType::FLOAT>(input_grad_accessor);

      CHECK(contains_non_zero(host_grad_input_data));
    }
  }

  TEST_CASE("Check Reverse Forward and Backward Kernels against CPU Kernels") {
    std::size_t num_out_blks = 2;
    std::size_t reverse_dim_size = 3;
    std::size_t in_blk_size = 5;

    TensorShape input_shape = make_tensor_shape_from_legion_dims(
        {num_out_blks, reverse_dim_size, in_blk_size}, DataType::FLOAT);
    TensorShape output_shape = input_shape;

    ManagedPerDeviceFFHandle managed_handle{};
    ManagedFFStream managed_stream{};

    Allocator gpu_allocator = create_local_cuda_memory_allocator();
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    SUBCASE("forward_kernel") {
      auto transform = [counter = 0.0f](float val) mutable {
        return counter++;
      };

      // Run GPU Cast Forward Kernel
      GenericTensorAccessorW input_accessor_gpu =
          create_transformed_accessor_w<float, float>(
              input_shape, gpu_allocator, transform);
      GenericTensorAccessorW output_accessor_gpu =
          gpu_allocator.allocate_tensor(output_shape);

      Kernels::Reverse::forward_kernel(managed_stream.raw_stream(),
                                       input_accessor_gpu.get_float_ptr(),
                                       output_accessor_gpu.get_float_ptr(),
                                       num_out_blks,
                                       reverse_dim_size,
                                       in_blk_size,
                                       input_accessor_gpu.shape.num_elements());

      std::vector<float> result_data_gpu =
          load_accessor_data<DataType::FLOAT>(output_accessor_gpu);

      // Run CPU Cast Forward Kernel
      GenericTensorAccessorW input_accessor_cpu =
          create_transformed_accessor_w<float, float>(
              input_shape, cpu_allocator, transform);
      GenericTensorAccessorW output_accessor_cpu =
          cpu_allocator.allocate_tensor(output_shape);

      Kernels::Reverse::cpu_forward_kernel(
          input_accessor_cpu.get_float_ptr(),
          output_accessor_cpu.get_float_ptr(),
          num_out_blks,
          reverse_dim_size,
          in_blk_size,
          input_accessor_cpu.shape.num_elements());

      std::vector<float> result_data_cpu =
          load_accessor_data<DataType::FLOAT>(output_accessor_cpu);

      CHECK(result_data_gpu == result_data_cpu);
    }

    SUBCASE("backward_kernel") {
      // Run GPU Cast Backward Kernel
      GenericTensorAccessorW output_grad_accessor_gpu =
          create_random_filled_accessor_w<DataType::FLOAT>(output_shape,
                                                           gpu_allocator);
      GenericTensorAccessorW input_grad_accessor_gpu =
          gpu_allocator.allocate_tensor(input_shape);

      Kernels::Reverse::backward_kernel(
          managed_stream.raw_stream(),
          output_grad_accessor_gpu.get_float_ptr(),
          input_grad_accessor_gpu.get_float_ptr(),
          num_out_blks,
          reverse_dim_size,
          in_blk_size,
          input_grad_accessor_gpu.shape.num_elements());

      std::vector<float> result_data_gpu =
          load_accessor_data<DataType::FLOAT>(input_grad_accessor_gpu);

      // Run CPU Cast Backward Kernel
      GenericTensorAccessorW output_grad_accessor_cpu =
          copy_tensor_between_memories<DataType::FLOAT>(
              read_only_accessor_from_write_accessor(output_grad_accessor_gpu),
              cpu_allocator);
      GenericTensorAccessorW input_grad_accessor_cpu =
          cpu_allocator.allocate_tensor(input_shape);

      Kernels::Reverse::cpu_backward_kernel(
          output_grad_accessor_cpu.get_float_ptr(),
          input_grad_accessor_cpu.get_float_ptr(),
          num_out_blks,
          reverse_dim_size,
          in_blk_size,
          input_grad_accessor_cpu.shape.num_elements());

      std::vector<float> result_data_cpu =
          load_accessor_data<DataType::FLOAT>(input_grad_accessor_cpu);

      CHECK(result_data_gpu == result_data_cpu);
    }
  }
}
