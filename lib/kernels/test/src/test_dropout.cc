#include "doctest/doctest.h"
#include "kernels/dropout_kernels.h"
#include "kernels/local_allocator.h"
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

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Dropout Forward and Backward Kernels") {
    std::size_t num_elements = 100;
    std::size_t dims[] = {10, 10};
    std::size_t num_dims = 2;
    float dropout_rate = 0.1;
    unsigned long long seed = 12345;
    ArrayShape shape(dims, num_dims);

    PerDeviceFFHandle handle;
    setPerDeviceFFHandle(&handle);

    Allocator allocator = get_local_memory_allocator();

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    DropoutPerDeviceState state = Kernels::Dropout::init_kernel(
        handle, dropout_rate, seed, shape, allocator);

    float *input_data, *output_data, *grad_input_data;
    std::vector<float **> ptrs = {&input_data, &output_data, &grad_input_data};
    std::vector<size_t> sizes = {num_elements, num_elements, num_elements};
    allocate_ptrs(ptrs, sizes, allocator);
    randomFillDeviceData(&input_data, num_elements);

    Kernels::Dropout::forward_kernel(stream, state, input_data, output_data);

    std::vector<float> host_output_data(num_elements, 0.0f);
    checkCUDA(cudaMemcpy(host_output_data.data(), output_data,
                         num_elements * sizeof(float), cudaMemcpyDeviceToHost));

    int zero_count = 0;
    for (float value : host_output_data) {
      if (value == 0.0f)
        zero_count++;
    }
    CHECK(zero_count ==
          doctest::Approx(num_elements * dropout_rate).epsilon(0.5));

    Kernels::Dropout::backward_kernel(stream, state, output_data,
                                      grad_input_data);

    std::vector<float> host_grad_input_data(num_elements);
    checkCUDA(cudaMemcpy(host_grad_input_data.data(), grad_input_data,
                         num_elements * sizeof(float), cudaMemcpyDeviceToHost));

    Kernels::Dropout::cleanup_kernel(allocator, state.inputTensor,
                                     state.outputTensor, state.dropoutDesc,
                                     state.dropoutStates);

    checkCUDA(cudaStreamDestroy(stream));
  }
}
