#include "doctest/doctest.h"
#include "kernels/local_allocator.h"
#include "kernels/pool_2d_kernels.h"
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
  TEST_CASE("Test Pool2D Forward and Backward Kernel") {
    int input_w = 10, input_h = 10, input_c = 3, input_n = 1;
    int output_w = 5, output_h = 5, output_c = 3, output_n = 1;
    int pad_h = 0, pad_w = 0, kernel_h = 2, kernel_w = 2, stride_h = 2,
        stride_w = 2;
    std::size_t num_elements = input_w * input_h * input_c * input_n;
    std::size_t output_elements = output_w * output_h * output_c * output_n;
    PoolOp pool_type = PoolOp::MAX;

    PerDeviceFFHandle handle;
    setPerDeviceFFHandle(&handle);
    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Allocator allocator = get_local_memory_allocator();

    float *input_data, *output_data;
    std::vector<float **> ptrs = {&input_data, &output_data};
    std::vector<size_t> sizes = {num_elements, output_elements};
    allocate_ptrs(ptrs, sizes, allocator);

    randomFillDeviceData(&input_data, num_elements);

    Pool2DPerDeviceState state = Kernels::Pool2D::init_kernel(
        handle, std::nullopt, input_w, input_h, input_c, input_n, output_w,
        output_h, output_c, output_n, pad_h, pad_w, kernel_h, kernel_w,
        stride_h, stride_w, pool_type);

    Kernels::Pool2D::forward_kernel(stream, state, input_data, output_data);

    std::vector<float> host_output_data(output_elements);
    checkCUDA(cudaMemcpy(host_output_data.data(), output_data,
                         output_elements * sizeof(float),
                         cudaMemcpyDeviceToHost));

    float *output_grad, *input_grad;
    std::vector<float **> ptrs_grad = {&output_grad, &input_grad};
    std::vector<size_t> sizes_grad = {output_elements, num_elements};
    allocate_ptrs(ptrs_grad, sizes_grad, allocator);
    fillDeviceDataNum(&output_grad, output_elements, 1.0f);

    Kernels::Pool2D::backward_kernel(stream, state, input_data, input_grad,
                                     output_data, output_grad);

    std::vector<float> host_input_grad(num_elements);
    checkCUDA(cudaMemcpy(host_input_grad.data(), input_grad,
                         num_elements * sizeof(float), cudaMemcpyDeviceToHost));

    checkCUDA(cudaStreamDestroy(stream));
    checkCUDA(cudaFree(handle.workSpace));
    cudnnDestroy(handle.dnn);
    cublasDestroy(handle.blas);
  }
}
