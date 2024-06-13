#include "doctest/doctest.h"
#include "kernels/local_allocator.h"
#include "kernels/replicate_kernels.h"
#include "test_utils.h"
#include <algorithm>
#include <random>
#include <vector>

template <typename T>
void allocate_ptrs(std::vector<T **> &gpu_data_ptrs,
                   std::vector<size_t> const &num_elements,
                   Allocator &allocator) {
  for (size_t i = 0; i < gpu_data_ptrs.size(); ++i) {
    *gpu_data_ptrs[i] =
        static_cast<T *>(allocator.allocate(num_elements[i] * sizeof(float)));
  }
}

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Replicate Forward") {
    std::size_t num_elements = 100;
    std::size_t dims[] = {num_elements};
    std::size_t num_dims = 1;
    FlexFlow::ArrayShape shape(dims, num_dims);

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Allocator allocator = get_local_memory_allocator();

    float *input_data, *output_data;
    std::vector<float **> ptrs = {&input_data, &output_data};
    std::vector<size_t> sizes = {num_elements, num_elements};
    allocate_ptrs(ptrs, sizes, allocator);

    fillDeviceDataNum(&input_data, num_elements, 1.0f);

    const GenericTensorAccessorR input_accessor{
        DataType::FLOAT, shape, input_data};
    const GenericTensorAccessorW forward_output_accessor{
        DataType::FLOAT, shape, output_data};

    Kernels::Replicate::forward_kernel(
        stream, input_accessor, forward_output_accessor);

    std::vector<float> check_output_data(num_elements);
    checkCUDA(cudaMemcpy(check_output_data.data(),
                         output_data,
                         num_elements * sizeof(float),
                         cudaMemcpyDeviceToHost));

    for (std::size_t i = 0; i < num_elements; ++i) {
      REQUIRE(1.0f == check_output_data[i]);
    }
    checkCUDA(cudaStreamDestroy(stream));
  }

  TEST_CASE("Test Replicate Backward Kernel") {
    std::size_t num_elements = 100;
    size_t num_replicas = 5;
    std::size_t dims[] = {num_elements};
    std::size_t num_dims = 1;
    ArrayShape shape(dims, num_dims);

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Allocator allocator = get_local_memory_allocator();

    float *replicated_data, *aggregated_data;
    std::vector<float **> ptrs = {&replicated_data, &aggregated_data};
    std::vector<size_t> sizes = {num_elements * num_replicas, num_elements};
    allocate_ptrs(ptrs, sizes, allocator);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> host_input_data(num_elements);
    for (float &val : host_input_data) {
      val = dist(gen);
    }

    for (size_t i = 0; i < num_replicas; ++i) {
      checkCUDA(cudaMemcpy(replicated_data + i * num_elements,
                           host_input_data.data(),
                           num_elements * sizeof(float),
                           cudaMemcpyHostToDevice));
    }

    const GenericTensorAccessorR input_accessor{
        DataType::FLOAT, shape, replicated_data};
    const GenericTensorAccessorW output_accessor{
        DataType::FLOAT, shape, aggregated_data};

    Kernels::Replicate::backward_kernel(
        stream, output_accessor, input_accessor, num_replicas);

    checkCUDA(cudaStreamDestroy(stream));
  }
}
