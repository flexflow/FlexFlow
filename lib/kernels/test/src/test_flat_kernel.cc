#include "doctest/doctest.h"
#include "kernels/flat_kernels.h"
#include "kernels/local_allocator.h"
#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

namespace FlexFlow {
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Flat Kernel Forward and Backward") {
    std::size_t num_elements = 100;
    std::size_t dims[] = {num_elements};
    std::size_t num_dims = 1;
    FlexFlow::ArrayShape shape(dims, num_dims);

    Allocator allocator = get_local_memory_allocator();

    float *input_data =
        static_cast<float *>(allocator.allocate(num_elements * sizeof(float)));
    const GenericTensorAccessorR input_accessor{DataType::FLOAT, shape,
                                                input_data};
    std::vector<float> host_input_data(num_elements, 2.0f);
    checkCUDA(cudaMemcpy(input_data, host_input_data.data(),
                         num_elements * sizeof(float), cudaMemcpyHostToDevice));

    float *output_data =
        static_cast<float *>(allocator.allocate(num_elements * sizeof(float)));

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Kernels::Flat::forward_kernel(stream, input_accessor, output_data);

    std::vector<float> check_output_data(num_elements);
    checkCUDA(cudaMemcpy(check_output_data.data(), output_data,
                         num_elements * sizeof(float), cudaMemcpyDeviceToHost));

    for (std::size_t i = 0; i < num_elements; ++i) {
      REQUIRE(host_input_data[i] == check_output_data[i]);
    }

    std::vector<float> host_output_data(num_elements, 1.0f);
    float *add_data =
        static_cast<float *>(allocator.allocate(num_elements * sizeof(float)));
    checkCUDA(cudaMemcpy(add_data, host_output_data.data(),
                         num_elements * sizeof(float), cudaMemcpyHostToDevice));
    const GenericTensorAccessorR data_accessor{DataType::FLOAT, shape,
                                               add_data};

    Kernels::Flat::backward_kernel(stream, input_accessor, output_data,
                                   add_data);

    std::vector<float> backward_output_data(num_elements);
    checkCUDA(cudaMemcpy(backward_output_data.data(), output_data,
                         num_elements * sizeof(float), cudaMemcpyDeviceToHost));

    for (std::size_t i = 0; i < num_elements; ++i) {
      CHECK(backward_output_data[i] == 3.0f);
    }
    checkCUDA(cudaStreamDestroy(stream));
  }
}
} // namespace FlexFlow
