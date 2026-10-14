#include "internal/test_utils.h"
#include "kernels/reshape_kernels_gpu.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;
TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("Test Reshape Forward and Backward") {
    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    TensorShape input_shape = TensorShape{
        TensorDims{FFOrdered{100_p}},
        DataType::FLOAT,
    };
    TensorShape output_shape = TensorShape{
        TensorDims{FFOrdered{100_p}},
        DataType::INT32,
    };

    SUBCASE("gpu_forward_kernel") {
      GenericTensorAccessorR input_accessor =
          create_random_filled_accessor_r(input_shape, allocator);
      GenericTensorAccessorW output_accessor =
          allocator.allocate_tensor(output_shape);

      reshape_gpu_forward_kernel(
          managed_stream.raw_stream(), input_accessor, output_accessor);

      CHECK(contains_non_zero(output_accessor));
    }

    SUBCASE("gpu_backward_kernel") {
      GenericTensorAccessorR output_grad_accessor =
          create_random_filled_accessor_r(output_shape, allocator);
      GenericTensorAccessorW input_grad_accessor =
          allocator.allocate_tensor(input_shape);

      reshape_gpu_backward_kernel(managed_stream.raw_stream(),
                                  output_grad_accessor,
                                  input_grad_accessor);

      CHECK(contains_non_zero(input_grad_accessor));
    }
  }
}
