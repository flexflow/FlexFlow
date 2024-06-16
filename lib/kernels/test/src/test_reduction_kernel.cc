#include "doctest/doctest.h"
#include "kernels/reduction_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Reduction Forward and Backward Kernel") {
    std::size_t num_elements = 10;
    std::size_t num_replicas = 10;
    std::size_t total_elements = num_elements * num_replicas;

    TensorShape shape = get_float_tensor_shape({num_elements});
    TensorShape expanded_shape = get_float_tensor_shape({total_elements});

    PerDeviceFFHandle handle;
    setPerDeviceFFHandle(&handle);
    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Allocator allocator = get_local_memory_allocator();

    SUBCASE("Test Reduction Forward") {
      GenericTensorAccessorR input_accessor = makeReadOnlyAccessor(
          getRandomFilledAccessorW(expanded_shape, allocator));
      GenericTensorAccessorW output_accessor =
          getRandomFilledAccessorW(expanded_shape, allocator);

      Kernels::Reduction::forward_kernel(
          stream, input_accessor, output_accessor, num_replicas);

      std::vector<float> host_output_data =
          fill_host_data<float>(output_accessor.ptr, num_elements);

      SUBCASE("Test Reduction Backward") {
        GenericTensorAccessorR grad_accessor = makeReadOnlyAccessor(
            getFilledAccessorW(expanded_shape, allocator, 1.0f));

        Kernels::Reduction::backward_kernel(
            stream, output_accessor, grad_accessor);

        std::vector<float> host_grad_data =
            fill_host_data<float>(output_accessor.ptr, total_elements);
      }
    }

    cleanup_test(stream, handle);
  }
}
