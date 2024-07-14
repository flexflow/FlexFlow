#include "doctest/doctest.h"
#include "kernels/cast_kernels.h"
#include "kernels/cast_kernels_cpu.h"
#include "test_utils.h"
#include <type_traits>

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Call Cast Forward and Backward Kernels") {
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    TensorShape input_shape =
        make_tensor_shape_from_legion_dims<DataType::FLOAT>({100, 100});
    TensorShape output_shape =
        make_tensor_shape_from_legion_dims<DataType::DOUBLE>({100, 100});

    GenericTensorAccessorW output_accessor =
        create_random_filled_accessor_w(output_shape, allocator);

    SUBCASE("forward_kernel") {
      GenericTensorAccessorR input_accessor =
          read_only_accessor_from_write_accessor(
              create_random_filled_accessor_w(input_shape, allocator));

      Kernels::Cast::forward_kernel(managed_stream.raw_stream(),
                                    input_accessor,
                                    output_accessor,
                                    DataType::FLOAT,
                                    DataType::DOUBLE);

      std::vector<double> host_double_data =
          load_accessor_data<DataType::DOUBLE>(
              read_only_accessor_from_write_accessor(output_accessor));

      CHECK(contains_non_zero(host_double_data));
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorW grad_input_accessor =
          allocator.allocate_tensor(input_shape);

      Kernels::Cast::backward_kernel(
          managed_stream.raw_stream(),
          read_only_accessor_from_write_accessor(output_accessor),
          grad_input_accessor,
          DataType::DOUBLE,
          DataType::FLOAT);

      std::vector<float> host_grad_float_data =
          load_accessor_data<DataType::FLOAT>(
              read_only_accessor_from_write_accessor(grad_input_accessor));
      CHECK(contains_non_zero(host_grad_float_data));
    }
  }

  TEST_CASE("Check Cast Forward Kernel against CPU Kernel") {
    ManagedFFStream managed_stream{};

    Allocator gpu_allocator = create_local_cuda_memory_allocator();
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    TensorShape input_shape =
        make_tensor_shape_from_legion_dims<DataType::FLOAT>({100, 100});
    TensorShape output_shape =
        make_tensor_shape_from_legion_dims<DataType::INT32>({100, 100});

    GenericTensorAccessorW output_accessor_gpu =
        gpu_allocator.allocate_tensor(output_shape);
    GenericTensorAccessorW output_accessor_cpu =
        cpu_allocator.allocate_tensor(output_shape);

    // Only calling forward kernel as backward kernel is exactly the same
    SUBCASE("forward_kernel") {
      auto transform = [start_val = 1.1f,
                        counter = 0.0f](float input) mutable -> float {
        return start_val + counter++;
      };

      // Run GPU Forward Kernel
      GenericTensorAccessorW input_accessor_gpu =
          create_transformed_accessor_w<float, float>(
              input_shape, gpu_allocator, transform, false);
      Kernels::Cast::forward_kernel(
          managed_stream.raw_stream(),
          read_only_accessor_from_write_accessor(input_accessor_gpu),
          output_accessor_gpu,
          DataType::FLOAT,
          DataType::INT32);
      std::vector<int32_t> result_data_gpu =
          load_accessor_data<DataType::INT32>(
              read_only_accessor_from_write_accessor(output_accessor_gpu),
              false);

      // Run CPU Forward Kernel
      GenericTensorAccessorW input_accessor_cpu =
          create_transformed_accessor_w<float, float>(
              input_shape, cpu_allocator, transform, true);
      Kernels::Cast::CPU::forward_kernel(
          read_only_accessor_from_write_accessor(input_accessor_cpu),
          output_accessor_cpu,
          DataType::FLOAT,
          DataType::INT32);
      std::vector<int32_t> result_data_cpu =
          load_accessor_data<DataType::INT32>(
              read_only_accessor_from_write_accessor(output_accessor_cpu),
              true);

      CHECK(result_data_gpu == result_data_cpu);
    }
  }
}
