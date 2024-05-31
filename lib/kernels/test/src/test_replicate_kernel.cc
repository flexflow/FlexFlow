#include "doctest/doctest.h"
#include "kernels/local_allocator.h"
#include "kernels/replicate_kernels.h"
#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

namespace FlexFlow {
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Replicate Forward") {
    std::size_t num_elements = 100;
    std::size_t dims[] = {num_elements};
    std::size_t num_dims = 1;
    FlexFlow::ArrayShape shape(dims, num_dims);

    Allocator allocator = get_local_memory_allocator();

    float *input_data =
        static_cast<float *>(allocator.allocate(num_elements * sizeof(float)));
    const GenericTensorAccessorR input_accessor{DataType::FLOAT, shape,
                                                input_data};
    std::vector<float> host_input_data(num_elements, 1.0f);
    checkCUDA(cudaMemcpy(input_data, host_input_data.data(),
                         num_elements * sizeof(float), cudaMemcpyHostToDevice));

    float *output_data =
        static_cast<float *>(allocator.allocate(num_elements * sizeof(float)));
    const GenericTensorAccessorW forward_output_accessor{DataType::FLOAT, shape,
                                                         output_data};
    std::vector<float> check_output_data(num_elements);

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Kernels::Replicate::forward_kernel(stream, input_accessor,
                                       forward_output_accessor);

    checkCUDA(cudaMemcpy(check_output_data.data(), output_data,
                         num_elements * sizeof(float), cudaMemcpyDeviceToHost));

    for (std::size_t i = 0; i < num_elements; ++i) {
      REQUIRE(host_input_data[i] == check_output_data[i]);
    }
  }

  TEST_CASE("Test Replicate Backward Kernel") {
    std::size_t num_elements = 100;
    size_t num_replicas = 5;
    std::size_t dims[] = {num_elements};
    std::size_t num_dims = 1;
    ArrayShape shape(dims, num_dims);

    Allocator allocator = get_local_memory_allocator();
    float *replicated_data = static_cast<float *>(
        allocator.allocate(num_elements * num_replicas * sizeof(float)));
    float *aggregated_data =
        static_cast<float *>(allocator.allocate(num_elements * sizeof(float)));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> host_input_data(num_elements);
    for (auto &val : host_input_data) {
      val = dist(gen);
    }

    for (size_t i = 0; i < num_replicas; ++i) {
      checkCUDA(cudaMemcpy(replicated_data + i * num_elements,
                           host_input_data.data(), num_elements * sizeof(float),
                           cudaMemcpyHostToDevice));
    }

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    const GenericTensorAccessorR input_accessor{DataType::FLOAT, shape,
                                                replicated_data};
    const GenericTensorAccessorW output_accessor{DataType::FLOAT, shape,
                                                 aggregated_data};

    Kernels::Replicate::backward_kernel(stream, output_accessor, input_accessor,
                                        num_replicas);

    std::vector<float> host_aggregated_data(num_elements);
    checkCUDA(cudaMemcpy(host_aggregated_data.data(), aggregated_data,
                         num_elements * sizeof(float), cudaMemcpyDeviceToHost));

    for (std::size_t i = 0; i < num_elements; ++i) {
      float expected_sum = host_input_data[i] * num_replicas;
      CHECK(host_aggregated_data[i] == doctest::Approx(expected_sum));
    }

    checkCUDA(cudaStreamDestroy(stream));
  }
}
} // namespace FlexFlow
