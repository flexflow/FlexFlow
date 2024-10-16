#include "doctest/doctest.h"
#include "kernels/cast_kernels.h"
#include "kernels/cast_kernels_cpu.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Call Cast Forward and Backward Kernels") {
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    TensorShape input_shape =
        make_tensor_shape_from_legion_dims({100, 100}, DataType::FLOAT);
    TensorShape output_shape =
        make_tensor_shape_from_legion_dims({100, 100}, DataType::DOUBLE);

    SUBCASE("forward_kernel") {
      GenericTensorAccessorR input_accessor =
          create_random_filled_accessor_r(input_shape, allocator);
      GenericTensorAccessorW output_accessor =
          allocator.allocate_tensor(output_shape);

      Kernels::Cast::forward_kernel(managed_stream.raw_stream(),
                                    input_accessor,
                                    output_accessor,
                                    DataType::FLOAT,
                                    DataType::DOUBLE);

      CHECK(contains_non_zero(output_accessor));
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorR grad_output_accessor =
          create_random_filled_accessor_r(output_shape, allocator);
      GenericTensorAccessorW grad_input_accessor =
          create_zero_filled_accessor_w(input_shape, allocator);

      Kernels::Cast::backward_kernel(managed_stream.raw_stream(),
                                     grad_output_accessor,
                                     grad_input_accessor,
                                     DataType::DOUBLE,
                                     DataType::FLOAT);

      CHECK(contains_non_zero(grad_input_accessor));
    }
  }

  TEST_CASE("Check Cast Forward Kernel against CPU Kernel") {
    ManagedFFStream managed_stream{};

    Allocator gpu_allocator = create_local_cuda_memory_allocator();
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    TensorShape input_shape =
        make_tensor_shape_from_legion_dims({10, 2}, DataType::FLOAT);
    TensorShape output_shape =
        make_tensor_shape_from_legion_dims({10, 2}, DataType::DOUBLE);

    // Only calling forward kernel as backward kernel is exactly the same
    SUBCASE("forward_kernel") {
      // Run GPU Forward Kernel
      GenericTensorAccessorR input_accessor_gpu =
          create_random_filled_accessor_r(input_shape, gpu_allocator);
      GenericTensorAccessorW output_accessor_gpu =
          create_zero_filled_accessor_w(output_shape, gpu_allocator);

      Kernels::Cast::forward_kernel(managed_stream.raw_stream(),
                                    input_accessor_gpu,
                                    output_accessor_gpu,
                                    DataType::FLOAT,
                                    DataType::DOUBLE);

      // Run CPU Forward Kernel
      GenericTensorAccessorR input_accessor_cpu =
          copy_tensor_accessor_r(input_accessor_gpu, cpu_allocator);
      GenericTensorAccessorW output_accessor_cpu =
          create_zero_filled_accessor_w(output_shape, cpu_allocator);

      Kernels::Cast::cpu_forward_kernel(input_accessor_cpu,
                                        output_accessor_cpu,
                                        DataType::FLOAT,
                                        DataType::DOUBLE);

      CHECK(w_accessors_are_equal<DataType::DOUBLE>(output_accessor_gpu,
                                                    output_accessor_cpu));
    }
  }
}
