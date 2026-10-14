#include "kernels/accessor_contains_non_zero_value.h"
#include "kernels/accessors_are_equal.h"
#include "kernels/allocation.h"
#include "kernels/cast_kernels.h"
#include "kernels/cast_kernels_cpu.h"
#include "kernels/cast_kernels_gpu.h"
#include "kernels/copy_tensor_accessor.h"
#include "kernels/create_random_filled_accessor.h"
#include "kernels/create_zero_filled_accessor.h"
#include "kernels/local_cpu_allocator.h"
#include "kernels/local_cuda_allocator.h"
#include "kernels/managed_ff_stream.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;
TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("Call Cast Forward and Backward Kernels") {
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    TensorShape input_shape = TensorShape{
        TensorDims{FFOrdered{100_p, 100_p}},
        DataType::FLOAT,
    };
    TensorShape output_shape = TensorShape{
        TensorDims{FFOrdered{100_p, 100_p}},
        DataType::DOUBLE,
    };

    SUBCASE("gpu_forward_kernel") {
      GenericTensorAccessorR input_accessor =
          create_random_filled_accessor_r(input_shape, allocator);
      GenericTensorAccessorW output_accessor =
          allocator.allocate_tensor(output_shape);

      Kernels::Cast::gpu_forward_kernel(
          managed_stream.raw_stream(), input_accessor, output_accessor);

      CHECK(contains_non_zero(output_accessor));
    }

    SUBCASE("gpu_backward_kernel") {
      GenericTensorAccessorR grad_output_accessor =
          create_random_filled_accessor_r(output_shape, allocator);
      GenericTensorAccessorW grad_input_accessor =
          create_zero_filled_accessor_w(input_shape, allocator);

      Kernels::Cast::gpu_backward_kernel(managed_stream.raw_stream(),
                                         grad_output_accessor,
                                         grad_input_accessor);

      CHECK(contains_non_zero(grad_input_accessor));
    }
  }

  TEST_CASE("Check Cast Forward Kernel against CPU Kernel") {
    ManagedFFStream managed_stream{};

    Allocator gpu_allocator = create_local_cuda_memory_allocator();
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    TensorShape input_shape = TensorShape{
        TensorDims{FFOrdered{10_p, 2_p}},
        DataType::FLOAT,
    };
    TensorShape output_shape = TensorShape{
        TensorDims{FFOrdered{10_p, 2_p}},
        DataType::DOUBLE,
    };

    // Only calling forward kernel as backward kernel is exactly the same
    SUBCASE("forward_kernel") {
      // Run GPU Forward Kernel
      GenericTensorAccessorR input_accessor_gpu =
          create_random_filled_accessor_r(input_shape, gpu_allocator);
      GenericTensorAccessorW output_accessor_gpu =
          create_zero_filled_accessor_w(output_shape, gpu_allocator);

      Kernels::Cast::gpu_forward_kernel(
          managed_stream.raw_stream(), input_accessor_gpu, output_accessor_gpu);

      // Run CPU Forward Kernel
      GenericTensorAccessorR input_accessor_cpu =
          copy_tensor_accessor_r(input_accessor_gpu, cpu_allocator);
      GenericTensorAccessorW output_accessor_cpu =
          create_zero_filled_accessor_w(output_shape, cpu_allocator);

      Kernels::Cast::cpu_forward_kernel(input_accessor_cpu,
                                        output_accessor_cpu);

      CHECK(accessors_are_equal(output_accessor_gpu, output_accessor_cpu));
    }
  }
}
