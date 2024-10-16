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
              create_filled_accessor_w(input_shape, allocator, 1.0f));
      GenericTensorAccessorW output_accessor =
          allocator.allocate_tensor(output_shape);

      Kernels::Reverse::forward_kernel(managed_stream.raw_stream(),
                                       input_accessor.get_float_ptr(),
                                       output_accessor.get_float_ptr(),
                                       num_out_blks,
                                       reverse_dim_size,
                                       in_blk_size,
                                       input_accessor.shape.num_elements());

      CHECK(contains_non_zero(output_accessor));
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorW output_grad_accessor =
          create_random_filled_accessor_w(output_shape, allocator);
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

      CHECK(contains_non_zero(input_grad_accessor));
    }
  }

  TEST_CASE("Check Reverse Forward and Backward Kernels against CPU Kernels") {
    std::size_t num_out_blks = 1;
    std::size_t reverse_dim_size = 3;
    std::size_t in_blk_size = 2;

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
      GenericTensorAccessorR input_accessor_gpu =
          create_random_filled_accessor_r(input_shape, gpu_allocator);
      GenericTensorAccessorW output_accessor_gpu =
          create_zero_filled_accessor_w(output_shape, gpu_allocator);

      Kernels::Reverse::forward_kernel(managed_stream.raw_stream(),
                                       input_accessor_gpu.get_float_ptr(),
                                       output_accessor_gpu.get_float_ptr(),
                                       num_out_blks,
                                       reverse_dim_size,
                                       in_blk_size,
                                       input_accessor_gpu.shape.num_elements());

      // Run CPU Cast Forward Kernel
      GenericTensorAccessorR input_accessor_cpu =
          copy_tensor_accessor_r(input_accessor_gpu, cpu_allocator);
      GenericTensorAccessorW output_accessor_cpu =
          create_zero_filled_accessor_w(output_shape, cpu_allocator);

      Kernels::Reverse::cpu_forward_kernel(input_accessor_cpu,
                                           output_accessor_cpu);

      CHECK(w_accessors_are_equal<DataType::FLOAT>(output_accessor_cpu,
                                                   output_accessor_cpu));
    }

    SUBCASE("backward_kernel") {
      // Run GPU Cast Backward Kernel
      GenericTensorAccessorR output_grad_accessor_gpu =
          create_random_filled_accessor_r(output_shape, gpu_allocator);
      GenericTensorAccessorW input_grad_accessor_gpu =
          create_zero_filled_accessor_w(input_shape, gpu_allocator);

      Kernels::Reverse::backward_kernel(
          managed_stream.raw_stream(),
          output_grad_accessor_gpu.get_float_ptr(),
          input_grad_accessor_gpu.get_float_ptr(),
          num_out_blks,
          reverse_dim_size,
          in_blk_size,
          input_grad_accessor_gpu.shape.num_elements());

      // Run CPU Cast Backward Kernel
      GenericTensorAccessorR output_grad_accessor_cpu =
          copy_tensor_accessor_r(output_grad_accessor_gpu, cpu_allocator);
      GenericTensorAccessorW input_grad_accessor_cpu =
          create_zero_filled_accessor_w(input_shape, cpu_allocator);

      Kernels::Reverse::cpu_backward_kernel(output_grad_accessor_cpu,
                                            input_grad_accessor_cpu);

      CHECK(w_accessors_are_equal<DataType::FLOAT>(input_grad_accessor_gpu,
                                                   input_grad_accessor_cpu));
    }
  }
}
