#include "doctest/doctest.h"
#include "kernels/flat_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Flat Kernel") {
    std::size_t num_elements = 100;

    TensorShape input_shape = get_float_tensor_shape({num_elements});

    Allocator allocator = get_local_memory_allocator();

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    GenericTensorAccessorR input_accessor =
        makeReadOnlyAccessor(getFilledAccessorW(input_shape, allocator, 2.0f));
    GenericTensorAccessorW output_accessor =
        allocator.allocate_tensor(input_shape);

    SUBCASE("Test flat kernel forward") {
      Kernels::Flat::forward_kernel(
          stream, input_accessor, (float *)output_accessor.ptr);

      std::vector<float> check_output_data =
          fill_host_data<float>(output_accessor.ptr, num_elements);

      for (std::size_t i = 0; i < num_elements; ++i) {
        REQUIRE(2.0f == check_output_data[i]);
      }
      SUBCASE("Test flat kernel backward") {
        GenericTensorAccessorR data_accessor = makeReadOnlyAccessor(
            getFilledAccessorW(input_shape, allocator, 1.0f));

        Kernels::Flat::backward_kernel(stream,
                                       input_accessor,
                                       (float *)output_accessor.ptr,
                                       (float const *)data_accessor.ptr);

        std::vector<float> backward_output_data =
            fill_host_data<float>(output_accessor.ptr, num_elements);

        for (std::size_t i = 0; i < num_elements; ++i) {
          CHECK(backward_output_data[i] == 3.0f);
        }
      }
    }

    checkCUDA(cudaStreamDestroy(stream));
  }
}
