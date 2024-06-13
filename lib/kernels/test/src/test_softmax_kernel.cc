#include "doctest/doctest.h"
#include "kernels/local_allocator.h"
#include "kernels/softmax_kernels.h"
#include "test_utils.h"
#include <cmath>
#include <numeric>
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
  TEST_CASE("Test Softmax Forward") {
    std::size_t num_elements = 100;
    int input_n = 1;
    int input_c = num_elements;
    int input_h = 1;
    int input_w = 1;

    PerDeviceFFHandle handle;
    setPerDeviceFFHandle(&handle);
    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Allocator allocator = get_local_memory_allocator();

    float *input_data, *output_data;
    std::vector<float **> ptrs = {&input_data, &output_data};
    std::vector<size_t> sizes = {num_elements, num_elements};
    allocate_ptrs(ptrs, sizes, allocator);

    std::vector<float> host_input_data =
        returnRandomFillDeviceData(&input_data, num_elements);

    SoftmaxPerDeviceState state = Kernels::Softmax::init_kernel(
        handle, 0, input_n, input_c, input_h, input_w);

    Kernels::Softmax::forward_kernel(stream, state, input_data, output_data);

    std::vector<float> host_output_data(num_elements);
    checkCUDA(cudaMemcpy(host_output_data.data(),
                         output_data,
                         num_elements * sizeof(float),
                         cudaMemcpyDeviceToHost));

    float max_input =
        *std::max_element(host_input_data.begin(), host_input_data.end());
    float sum_exp = std::accumulate(host_input_data.begin(),
                                    host_input_data.end(),
                                    0.0f,
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
    setPerDeviceFFHandle(&handle);
    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    SoftmaxPerDeviceState state = Kernels::Softmax::init_kernel(
        handle, 0, input_n, input_c, input_h, input_w);

    Allocator allocator = get_local_memory_allocator();
    float *input_data, *output_data;
    std::vector<float **> ptrs = {&input_data, &output_data};
    std::vector<size_t> sizes = {num_elements, num_elements};
    allocate_ptrs(ptrs, sizes, allocator);

    fillDeviceDataNum(&output_data, num_elements, 1.0f);

    Kernels::Softmax::backward_kernel(
        stream, input_data, output_data, num_elements);

    std::vector<float> check_output_data(num_elements);
    checkCUDA(cudaMemcpy(check_output_data.data(),
                         input_data,
                         num_elements * sizeof(float),
                         cudaMemcpyDeviceToHost));

    for (std::size_t i = 0; i < num_elements; ++i) {
      REQUIRE(1.0f == check_output_data[i]);
    }

    checkCUDA(cudaStreamDestroy(stream));
  }
}
