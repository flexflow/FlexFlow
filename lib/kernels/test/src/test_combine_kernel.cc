#include "doctest/doctest.h"
#include "kernels/combine_kernels.h"
#include "kernels/combine_kernels_cpu.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Call Combine Forward and Backward Kernels") {
    ManagedPerDeviceFFHandle managed_handle{};
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    TensorShape input_shape =
        make_tensor_shape_from_legion_dims({100, 100}, DataType::FLOAT);
    TensorShape output_shape = input_shape;

    SUBCASE("forward_kernel") {
      GenericTensorAccessorR input_accessor =
          create_random_filled_accessor_r<DataType::FLOAT>(input_shape,
                                                           allocator);
      GenericTensorAccessorW output_accessor =
          allocator.allocate_tensor(output_shape);

      Kernels::Combine::forward_kernel(
          managed_stream.raw_stream(), input_accessor, output_accessor);

      std::vector<float> host_output_data =
          load_accessor_data<DataType::FLOAT>(output_accessor);
      CHECK(contains_non_zero(host_output_data));
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorR output_grad_accessor =
          create_random_filled_accessor_r<DataType::FLOAT>(output_shape,
                                                           allocator);
      GenericTensorAccessorW input_grad_accessor =
          allocator.allocate_tensor(input_shape);

      Kernels::Combine::backward_kernel(managed_stream.raw_stream(),
                                        output_grad_accessor,
                                        input_grad_accessor);

      std::vector<float> host_input_grad =
          load_accessor_data<DataType::FLOAT>(input_grad_accessor);
      CHECK(contains_non_zero(host_input_grad));
    }
  }

  TEST_CASE("Check Combine Forward Kernel against CPU Kernel") {
    ManagedFFStream managed_stream{};

    Allocator gpu_allocator = create_local_cuda_memory_allocator();
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    TensorShape input_shape =
        make_tensor_shape_from_legion_dims({5, 5}, DataType::FLOAT);
    TensorShape output_shape = input_shape;

    SUBCASE("forward_kernel") {
      // Run GPU Combine Forward Kernel
      GenericTensorAccessorR input_accessor_gpu =
          create_random_filled_accessor_r<DataType::FLOAT>(input_shape,
                                                           gpu_allocator);
      GenericTensorAccessorW output_accessor_gpu =
          gpu_allocator.allocate_tensor(output_shape);

      Kernels::Combine::forward_kernel(
          managed_stream.raw_stream(), input_accessor_gpu, output_accessor_gpu);

      std::vector<float> result_data_gpu =
          load_accessor_data<DataType::FLOAT>(output_accessor_gpu);

      // Run CPU Combine Forward Kernel
      GenericTensorAccessorW input_accessor_cpu =
          copy_tensor_between_memories<DataType::FLOAT>(input_accessor_gpu,
                                                        cpu_allocator);
      GenericTensorAccessorW output_accessor_cpu =
          cpu_allocator.allocate_tensor(output_shape);

      Kernels::Combine::cpu_forward_kernel(
          read_only_accessor_from_write_accessor(input_accessor_cpu),
          output_accessor_cpu);

      std::vector<float> result_data_cpu =
          load_accessor_data<DataType::FLOAT>(output_accessor_cpu);

      CHECK(result_data_gpu == result_data_cpu);
    }

    SUBCASE("backward_kernel") {
      // Run GPU Combine Backward Kernel
      GenericTensorAccessorR output_grad_accessor_gpu =
          create_random_filled_accessor_r<DataType::FLOAT>(output_shape,
                                                           gpu_allocator);
      GenericTensorAccessorW input_grad_accessor_gpu =
          gpu_allocator.allocate_tensor_and_zero(input_shape);

      Kernels::Combine::backward_kernel(managed_stream.raw_stream(),
                                        output_grad_accessor_gpu,
                                        input_grad_accessor_gpu);

      std::vector<float> result_data_gpu =
          load_accessor_data<DataType::FLOAT>(input_grad_accessor_gpu);

      // Run CPU Combine Backward Kernel
      GenericTensorAccessorW output_grad_accessor_cpu =
          copy_tensor_between_memories<DataType::FLOAT>(
              output_grad_accessor_gpu, cpu_allocator);
      GenericTensorAccessorW input_grad_accessor_cpu =
          cpu_allocator.allocate_tensor_and_zero(input_shape);

      Kernels::Combine::cpu_backward_kernel(
          read_only_accessor_from_write_accessor(output_grad_accessor_cpu),
          input_grad_accessor_cpu);

      std::vector<float> result_data_cpu =
          load_accessor_data<DataType::FLOAT>(input_grad_accessor_cpu);

      CHECK(result_data_gpu == result_data_cpu);
    }
  }
}
