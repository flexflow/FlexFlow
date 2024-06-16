#include "doctest/doctest.h"
#include "kernels/replicate_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Replicate Kernel") {
    std::size_t num_elements = 100;
    std::size_t num_replicas = 10;

    TensorShape shape = get_float_tensor_shape({num_elements});

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Allocator allocator = get_local_memory_allocator();

    SUBCASE("Test Replicate Forward") {
      GenericTensorAccessorR input_accessor =
          makeReadOnlyAccessor(getFilledAccessorW(shape, allocator, 1.0f));
      GenericTensorAccessorW output_accessor =
          getFilledAccessorW(shape, allocator, 0.0f);

      Kernels::Replicate::forward_kernel(
          stream, input_accessor, output_accessor);

      std::vector<float> check_output_data =
          fill_host_data<float>(output_accessor.ptr, num_elements);

      for (std::size_t i = 0; i < num_elements; ++i) {
        REQUIRE(1.0f == check_output_data[i]);
      }

      SUBCASE("Test Replicate Backward") {
        GenericTensorAccessorR replicated_accessor = makeReadOnlyAccessor(
            getFilledAccessorW(shape, allocator, num_replicas));
        GenericTensorAccessorW aggregated_accessor =
            getFilledAccessorW(shape, allocator, 0.0f);

        Kernels::Replicate::backward_kernel(
            stream, aggregated_accessor, replicated_accessor, num_replicas);

        std::vector<float> check_aggregated_data =
            fill_host_data<float>(aggregated_accessor.ptr, num_elements);
        REQUIRE(contains_non_zero(check_aggregated_data));
      }
    }

    checkCUDA(cudaStreamDestroy(stream));
  }
}
