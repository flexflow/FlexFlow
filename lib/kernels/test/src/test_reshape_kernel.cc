#include "doctest/doctest.h"
#include "kernels/local_allocator.h"
#include "kernels/reshape_kernels.h"
#include "test_utils.h"
#include <algorithm>
#include <iostream>
#include <vector>

template <typename T>
void allocate_ptrs(std::vector<T **> &gpu_data_ptrs,
                   const std::vector<size_t> &num_elements,
                   Allocator &allocator) {
  for (size_t i = 0; i < gpu_data_ptrs.size(); ++i) {
    *gpu_data_ptrs[i] =
        static_cast<T *>(allocator.allocate(num_elements[i] * sizeof(float)));
  }
}

namespace FlexFlow {
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Reshape Forward and Backward") {
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

    const GenericTensorAccessorR input_accessor{DataType::FLOAT, shape,
                                                input_data};
    const GenericTensorAccessorW forward_output_accessor{DataType::FLOAT, shape,
                                                         output_data};

    ReshapePerDeviceState state =
        Kernels::Reshape::init_kernel(DataType::FLOAT);

    Kernels::Reshape::forward_kernel(stream, state, input_accessor,
                                     forward_output_accessor);

    std::vector<float> check_output_data(num_elements);
    checkCUDA(cudaMemcpy(check_output_data.data(), output_data,
                         num_elements * sizeof(float), cudaMemcpyDeviceToHost));

    for (std::size_t i = 0; i < num_elements; ++i) {
      REQUIRE(1.0f == check_output_data[i]);
    }

    float *grad_data =
        static_cast<float *>(allocator.allocate(num_elements * sizeof(float)));
    fillDeviceDataNum(&grad_data, num_elements, 1.0f);
    const GenericTensorAccessorR grad_accessor{DataType::FLOAT, shape,
                                               grad_data};

    Kernels::Reshape::backward_kernel(stream, state, forward_output_accessor,
                                      grad_accessor);

    std::vector<float> host_grad_input_data(num_elements);
    checkCUDA(cudaMemcpy(host_grad_input_data.data(), output_data,
                         num_elements * sizeof(float), cudaMemcpyDeviceToHost));

    for (std::size_t i = 0; i < num_elements; ++i) {
      CHECK(host_grad_input_data[i] == 2.0f);
    }
    
    checkCUDA(cudaStreamDestroy(stream));
  }
}
} // namespace FlexFlow
