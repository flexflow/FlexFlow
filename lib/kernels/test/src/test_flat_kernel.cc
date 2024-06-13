#include "doctest/doctest.h"
#include "kernels/flat_kernels.h"
#include "kernels/local_allocator.h"
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
  TEST_CASE("Test Flat Kernel Forward and Backward") {
    std::size_t num_elements = 100;
    std::size_t dims[] = {num_elements};
    std::size_t num_dims = 1;
    FlexFlow::ArrayShape shape(dims, num_dims);

    Allocator allocator = get_local_memory_allocator();

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    float *input_data, *output_data;
    std::vector<float **> ptrs = {&input_data, &output_data};
    std::vector<size_t> sizes = {num_elements, num_elements};
    allocate_ptrs(ptrs, sizes, allocator);
    fillDeviceDataNum(&input_data, num_elements, 2.0f);

    const GenericTensorAccessorR input_accessor{
        DataType::FLOAT, shape, input_data};

    Kernels::Flat::forward_kernel(stream, input_accessor, output_data);

    std::vector<float> check_output_data(num_elements);
    checkCUDA(cudaMemcpy(check_output_data.data(),
                         output_data,
                         num_elements * sizeof(float),
                         cudaMemcpyDeviceToHost));

    for (std::size_t i = 0; i < num_elements; ++i) {
      REQUIRE(2.0f == check_output_data[i]);
    }

    float *add_data =
        static_cast<float *>(allocator.allocate(num_elements * sizeof(float)));
    fillDeviceDataNum(&add_data, num_elements, 1.0f);
    const GenericTensorAccessorR data_accessor{
        DataType::FLOAT, shape, add_data};

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
    checkCUDA(cudaStreamDestroy(stream));
  }
}
