#include "doctest/doctest.h"
#include "kernels/local_allocator.h"
#include "kernels/softmax_kernels.h"
#include <cmath>
#include <numeric>
#include <vector>

namespace FlexFlow {
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Softmax Forward") {
    std::size_t num_elements = 100;

    std::vector<float> host_input_data(num_elements);
    for (auto &val : host_input_data) {
      val = static_cast<float>(rand()) / RAND_MAX;
    }

    int input_n = 1;
    int input_c = num_elements;
    int input_h = 1;
    int input_w = 1;

    PerDeviceFFHandle handle;
    cudnnCreate(&handle.dnn);
    cublasCreate(&handle.blas);
    handle.workSpaceSize = 1024 * 1024;
    cudaMalloc(&handle.workSpace, handle.workSpaceSize);
    handle.allowTensorOpMathConversion = true;

    SoftmaxPerDeviceState state = Kernels::Softmax::init_kernel(
        handle, 0, input_n, input_c, input_h, input_w);

    Allocator allocator = get_local_memory_allocator();
    float *input_data =
        static_cast<float *>(allocator.allocate(num_elements * sizeof(float)));
    float *output_data =
        static_cast<float *>(allocator.allocate(num_elements * sizeof(float)));

    checkCUDA(cudaMemcpy(input_data, host_input_data.data(),
                         num_elements * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Kernels::Softmax::forward_kernel(stream, state, input_data, output_data);

    std::vector<float> host_output_data(num_elements);
    checkCUDA(cudaMemcpy(host_output_data.data(), output_data,
                         num_elements * sizeof(float), cudaMemcpyDeviceToHost));

    float max_input =
        *std::max_element(host_input_data.begin(), host_input_data.end());
    float sum_exp =
        std::accumulate(host_input_data.begin(), host_input_data.end(), 0.0f,
                        [max_input](float acc, float val) {
                          return acc + std::exp(val - max_input);
                        });

    for (std::size_t i = 0; i < num_elements; ++i) {
      float expected_value = std::exp(host_input_data[i] - max_input) / sum_exp;
      CHECK(doctest::Approx(host_output_data[i]).epsilon(0.001) ==
            expected_value);
    }

    checkCUDA(cudaStreamDestroy(stream));
  }

  TEST_CASE("Test Softmax Backward") {
    std::size_t num_elements = 100;

    int input_n = 1;
    int input_c = 1;
    int input_h = 1;
    int input_w = num_elements;

    PerDeviceFFHandle handle;
    cudnnCreate(&handle.dnn);
    cublasCreate(&handle.blas);
    handle.workSpaceSize = 1024 * 1024;
    cudaMalloc(&handle.workSpace, handle.workSpaceSize);
    handle.allowTensorOpMathConversion = true;

    SoftmaxPerDeviceState state = Kernels::Softmax::init_kernel(
        handle, 0, input_n, input_c, input_h, input_w);

    Allocator allocator = get_local_memory_allocator();
    float *input_data =
        static_cast<float *>(allocator.allocate(num_elements * sizeof(float)));
    float *output_data =
        static_cast<float *>(allocator.allocate(num_elements * sizeof(float)));

    std::vector<float> host_input_data(num_elements);
    std::vector<float> host_output_data(num_elements, 1.0f);
    checkCUDA(cudaMemcpy(output_data, host_output_data.data(),
                         num_elements * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Kernels::Softmax::backward_kernel(stream, input_data, output_data,
                                      num_elements);

    std::vector<float> check_output_data(num_elements);
    checkCUDA(cudaMemcpy(check_output_data.data(), input_data,
                         num_elements * sizeof(float), cudaMemcpyDeviceToHost));

    for (std::size_t i = 0; i < num_elements; ++i) {
      REQUIRE(host_output_data[i] == check_output_data[i]);
    }

    checkCUDA(cudaStreamDestroy(stream));
  }
}
} // namespace FlexFlow
