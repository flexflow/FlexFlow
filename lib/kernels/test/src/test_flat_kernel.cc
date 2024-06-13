#include "doctest/doctest.h"
#include "kernels/flat_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Flat Kernel") {
    std::size_t num_elements = 100;
    ArrayShape shape = ArrayShape{
        std::vector<size_t>{num_elements},
    };

    Allocator allocator = get_local_memory_allocator();

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    float *input_data, *output_data;
    std::vector<float **> ptrs = {&input_data, &output_data};
    std::vector<size_t> sizes = {num_elements, num_elements};
    allocate_ptrs(ptrs, sizes, allocator);
    fillDeviceDataNum(&input_data, num_elements, 2.0f);
    GenericTensorAccessorR input_accessor{DataType::FLOAT, shape, input_data};

    SUBCASE("Test flat kernel forward") {
      Kernels::Flat::forward_kernel(stream, input_accessor, output_data);

      std::vector<float> check_output_data(num_elements);
      checkCUDA(cudaMemcpy(check_output_data.data(),
                           output_data,
                           num_elements * sizeof(float),
                           cudaMemcpyDeviceToHost));

      for (std::size_t i = 0; i < num_elements; ++i) {
        REQUIRE(2.0f == check_output_data[i]);
      }
    }

    SUBCASE("Test flat kernel backward") {
      float *add_data = static_cast<float *>(
          allocator.allocate(num_elements * sizeof(float)));
      fillDeviceDataNum(&add_data, num_elements, 1.0f);
      GenericTensorAccessorR data_accessor{DataType::FLOAT, shape, add_data};

      Kernels::Flat::backward_kernel(
          stream, input_accessor, output_data, add_data);

      std::vector<float> backward_output_data(num_elements);
      checkCUDA(cudaMemcpy(backward_output_data.data(),
                           output_data,
                           num_elements * sizeof(float),
                           cudaMemcpyDeviceToHost));

      for (std::size_t i = 0; i < num_elements; ++i) {
        CHECK(backward_output_data[i] == 3.0f);
      }
    }

    checkCUDA(cudaStreamDestroy(stream));
  }
}
