#include "doctest/doctest.h"
#include "kernels/replicate_kernels.h"
#include "kernels/replicate_kernels_cpu.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Call Replicate Forward and Backward Kernels") {
    std::size_t num_replicas = 10;

    TensorShape input_shape =
        make_tensor_shape_from_legion_dims({100}, DataType::FLOAT);
    TensorShape output_shape =
        make_tensor_shape_from_legion_dims({100}, DataType::FLOAT);

    ManagedPerDeviceFFHandle managed_handle{};
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    SUBCASE("forward_kernel") {
      GenericTensorAccessorR input_accessor =
          read_only_accessor_from_write_accessor(
              create_filled_accessor_w<float>(input_shape, allocator, 1.0f));
      GenericTensorAccessorW output_accessor =
          allocator.allocate_tensor(output_shape);

      Kernels::Replicate::forward_kernel(
          managed_stream.raw_stream(), input_accessor, output_accessor);

      std::vector<float> check_output_data =
          load_accessor_data<DataType::FLOAT>(output_accessor);

      std::vector<float> expected_output_data(
          input_accessor.shape.num_elements(), 1.0f);
      CHECK(vectors_are_approx_equal(check_output_data, expected_output_data));
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorW input_grad_accessor =
          create_filled_accessor_w<float>(input_shape, allocator, 1.0f);
      GenericTensorAccessorR output_grad_accessor =
          read_only_accessor_from_write_accessor(
              create_filled_accessor_w<float>(output_shape, allocator, 1.0f));

      Kernels::Replicate::backward_kernel(managed_stream.raw_stream(),
                                          input_grad_accessor,
                                          output_grad_accessor,
                                          num_replicas);

      std::vector<float> check_aggregated_data =
          load_accessor_data<DataType::FLOAT>(input_grad_accessor);
      CHECK(contains_non_zero(check_aggregated_data));
    }
  }

  TEST_CASE("Check Replicate Forward Kernel against CPU Kernel") {
    std::size_t num_replicas = 2;

    TensorShape input_shape =
        make_tensor_shape_from_legion_dims({5}, DataType::FLOAT);
    TensorShape output_shape =
        make_tensor_shape_from_legion_dims({5, num_replicas}, DataType::FLOAT);

    ManagedPerDeviceFFHandle managed_handle{};
    ManagedFFStream managed_stream{};

    Allocator gpu_allocator = create_local_cuda_memory_allocator();
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    SUBCASE("forward_kernel") {
      // Run GPU Replicate Forward Kernel
      GenericTensorAccessorR input_accessor_gpu =
          create_random_filled_accessor_r<DataType::FLOAT>(input_shape,
                                                           gpu_allocator);
      GenericTensorAccessorW output_accessor_gpu =
          gpu_allocator.allocate_tensor(output_shape);
      fill_with_zeros(output_accessor_gpu);

      Kernels::Replicate::forward_kernel(
          managed_stream.raw_stream(), input_accessor_gpu, output_accessor_gpu);

      std::vector<float> result_data_gpu =
          load_accessor_data<DataType::FLOAT>(output_accessor_gpu);

      // Run CPU Replicate Forward Kernel
      GenericTensorAccessorR input_accessor_cpu =
          copy_tensor_accessor_r(input_accessor_gpu, cpu_allocator);
      GenericTensorAccessorW output_accessor_cpu =
          cpu_allocator.allocate_tensor(output_shape);
      fill_with_zeros(output_accessor_cpu);

      Kernels::Replicate::cpu_forward_kernel(input_accessor_cpu,
                                             output_accessor_cpu);

      std::vector<float> result_data_cpu =
          load_accessor_data<DataType::FLOAT>(output_accessor_cpu);

      CHECK(vectors_are_approx_equal(result_data_gpu, result_data_cpu));
    }

    SUBCASE("backward_kernel") {
      // Run GPU Replicate Backward Kernel
      GenericTensorAccessorR output_grad_accessor_gpu =
          create_random_filled_accessor_r<DataType::FLOAT>(output_shape,
                                                           gpu_allocator);
      GenericTensorAccessorW input_grad_accessor_gpu =
          gpu_allocator.allocate_tensor(input_shape);
      fill_with_zeros(input_grad_accessor_gpu);

      Kernels::Replicate::backward_kernel(managed_stream.raw_stream(),
                                          input_grad_accessor_gpu,
                                          output_grad_accessor_gpu,
                                          num_replicas);

      std::vector<float> result_data_gpu =
          load_accessor_data<DataType::FLOAT>(input_grad_accessor_gpu);

      // Run CPU Replicate Backward Kernel
      GenericTensorAccessorR output_grad_accessor_cpu =
          copy_tensor_accessor_r(output_grad_accessor_gpu, cpu_allocator);

      GenericTensorAccessorW input_grad_accessor_cpu =
          cpu_allocator.allocate_tensor(input_shape);
      fill_with_zeros(input_grad_accessor_cpu);

      Kernels::Replicate::cpu_backward_kernel(
          input_grad_accessor_cpu, output_grad_accessor_cpu, num_replicas);

      std::vector<float> result_data_cpu =
          load_accessor_data<DataType::FLOAT>(input_grad_accessor_cpu);

      CHECK(vectors_are_approx_equal(result_data_gpu, result_data_cpu));
    }
  }
}
