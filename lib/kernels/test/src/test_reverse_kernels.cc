#include "kernels/accessor_contains_non_zero_value.h"
#include "kernels/accessors_are_equal.h"
#include "kernels/copy_tensor_accessor.h"
#include "kernels/create_constant_filled_accessor_r.h"
#include "kernels/create_random_filled_accessor.h"
#include "kernels/create_zero_filled_accessor.h"
#include "kernels/local_cpu_allocator.h"
#include "kernels/local_cuda_allocator.h"
#include "kernels/managed_ff_stream.h"
#include "kernels/managed_per_device_ff_handle.h"
#include "kernels/reverse_kernels_cpu.h"
#include "kernels/reverse_kernels_gpu.h"
#include "op-attrs/datatype_value.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;
TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("Call Reverse Forward and Backward Kernels") {
    TensorShape input_shape = TensorShape{
        TensorDims{FFOrdered{1_p, 10_p, 10_p}},
        DataType::FLOAT,
    };
    TensorShape output_shape = input_shape;

    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    ReverseAttrs attrs = ReverseAttrs{
        /*axis=*/ff_dim_t{0_n},
    };

    SUBCASE("gpu_forward_kernel") {
      GenericTensorAccessorR input_accessor =
          read_only_accessor_from_write_accessor(
              create_constant_filled_accessor_w(
                  input_shape, allocator, make_float_data_type_value(1)));
      GenericTensorAccessorW output_accessor =
          allocator.allocate_tensor(output_shape);

      Kernels::Reverse::gpu_forward_kernel(
          managed_stream.raw_stream(), input_accessor, output_accessor, attrs);

      CHECK(contains_non_zero(output_accessor));
    }

    SUBCASE("gpu_backward_kernel") {
      GenericTensorAccessorW output_grad_accessor =
          create_random_filled_accessor_w(output_shape, allocator);
      GenericTensorAccessorW input_grad_accessor =
          allocator.allocate_tensor(input_shape);

      Kernels::Reverse::gpu_backward_kernel(managed_stream.raw_stream(),
                                            output_grad_accessor,
                                            input_grad_accessor,
                                            attrs);

      CHECK(contains_non_zero(input_grad_accessor));
    }
  }

  TEST_CASE("Check Reverse Forward and Backward Kernels against CPU Kernels") {
    TensorShape input_shape = TensorShape{
        TensorDims{FFOrdered{1_p, 4_p, 3_p}},
        DataType::FLOAT,
    };
    TensorShape output_shape = input_shape;

    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);
    ManagedFFStream managed_stream{};

    Allocator gpu_allocator = create_local_cuda_memory_allocator();
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    ReverseAttrs attrs = ReverseAttrs{
        /*axis=*/ff_dim_t{0_n},
    };

    SUBCASE("gpu_forward_kernel") {
      // Run GPU Cast Forward Kernel
      GenericTensorAccessorR input_accessor_gpu =
          create_random_filled_accessor_r(input_shape, gpu_allocator);
      GenericTensorAccessorW output_accessor_gpu =
          create_zero_filled_accessor_w(output_shape, gpu_allocator);

      Kernels::Reverse::gpu_forward_kernel(managed_stream.raw_stream(),
                                           input_accessor_gpu,
                                           output_accessor_gpu,
                                           attrs);

      // Run CPU Cast Forward Kernel
      GenericTensorAccessorR input_accessor_cpu =
          copy_tensor_accessor_r(input_accessor_gpu, cpu_allocator);
      GenericTensorAccessorW output_accessor_cpu =
          create_zero_filled_accessor_w(output_shape, cpu_allocator);

      Kernels::Reverse::cpu_forward_kernel(
          input_accessor_cpu, output_accessor_cpu, attrs);

      CHECK(accessors_are_equal(output_accessor_cpu, output_accessor_cpu));
    }

    SUBCASE("gpu_backward_kernel") {
      // Run GPU Cast Backward Kernel
      GenericTensorAccessorR output_grad_accessor_gpu =
          create_random_filled_accessor_r(output_shape, gpu_allocator);

      GenericTensorAccessorW input_grad_accessor_gpu =
          create_zero_filled_accessor_w(input_shape, gpu_allocator);

      Kernels::Reverse::gpu_backward_kernel(managed_stream.raw_stream(),
                                            output_grad_accessor_gpu,
                                            input_grad_accessor_gpu,
                                            attrs);

      // Run CPU Cast Backward Kernel
      GenericTensorAccessorR output_grad_accessor_cpu =
          copy_tensor_accessor_r(output_grad_accessor_gpu, cpu_allocator);
      GenericTensorAccessorW input_grad_accessor_cpu =
          create_zero_filled_accessor_w(input_shape, cpu_allocator);

      Kernels::Reverse::cpu_backward_kernel(
          output_grad_accessor_cpu, input_grad_accessor_cpu, attrs);

      CHECK(accessors_are_equal(input_grad_accessor_gpu,
                                input_grad_accessor_cpu));
    }
  }
}
