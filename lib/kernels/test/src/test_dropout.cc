#include "doctest/doctest.h"
#include "kernels/dropout_kernels.h"
#include "kernels/local_allocator.h"
#include <algorithm>
#include <iostream>
#include <vector>

namespace FlexFlow {
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Dropout Forward and Backward Kernels") {
    std::size_t num_elements = 100;
    std::size_t dims[] = {10, 10};
    std::size_t num_dims = 2;
    float dropout_rate = 0.1;
    unsigned long long seed = 12345;
    ArrayShape shape(dims, num_dims);

    PerDeviceFFHandle handle;
    cudnnCreate(&handle.dnn);
    cublasCreate(&handle.blas);
    handle.workSpaceSize = 1024 * 1024;
    cudaMalloc(&handle.workSpace, handle.workSpaceSize);
    handle.allowTensorOpMathConversion = true;

    Allocator allocator = get_local_memory_allocator();
    DropoutPerDeviceState state = Kernels::Dropout::init_kernel(
        handle, dropout_rate, seed, shape, allocator);

    float *input_data =
        static_cast<float *>(allocator.allocate(num_elements * sizeof(float)));
    float *output_data =
        static_cast<float *>(allocator.allocate(num_elements * sizeof(float)));
    float *grad_input_data =
        static_cast<float *>(allocator.allocate(num_elements * sizeof(float)));

    std::vector<float> host_input_data(num_elements);
    std::generate(host_input_data.begin(), host_input_data.end(),
                  []() { return static_cast<float>(rand()) / RAND_MAX; });
    checkCUDA(cudaMemcpy(input_data, host_input_data.data(),
                         num_elements * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<float> host_output_data(num_elements, 0.0f);

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    // Forward kernel execution
    Kernels::Dropout::forward_kernel(stream, state, input_data, output_data);
    checkCUDA(cudaMemcpy(host_output_data.data(), output_data,
                         num_elements * sizeof(float), cudaMemcpyDeviceToHost));

    int zero_count = 0;
    for (auto value : host_output_data) {
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
} // namespace FlexFlow
