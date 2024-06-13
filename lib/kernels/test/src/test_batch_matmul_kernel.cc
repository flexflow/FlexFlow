#include "doctest/doctest.h"
#include "kernels/batch_matmul_kernels.h"
#include "kernels/local_allocator.h"
#include "test_utils.h"
#include <vector>

void allocate_ptrs(std::vector<float **> &gpu_data_ptrs,
                   std::vector<size_t> const &num_elements,
                   Allocator &allocator) {
  for (size_t i = 0; i < gpu_data_ptrs.size(); ++i) {
    *gpu_data_ptrs[i] = static_cast<float *>(
        allocator.allocate(num_elements[i] * sizeof(float)));
  }
}

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test BatchMatmul Forward and Backward Kernel") {
    int m = 10;
    int n = 10;
    int k = 10;
    int batch = 5;
    int a_seq_length_dim = -1;
    int b_seq_length_dim = -1;
    int seq_length = -1;

    size_t num_elements_a = m * k * batch;
    size_t num_elements_b = k * n * batch;
    size_t num_elements_output = m * n * batch;

    PerDeviceFFHandle handle;
    setPerDeviceFFHandle(&handle);
    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Allocator allocator = get_local_memory_allocator();

    float *a_input, *b_input, *output;
    std::vector<float **> ptrs = {&a_input, &b_input, &output};
    std::vector<size_t> sizes = {
        num_elements_a, num_elements_b, num_elements_output};

    allocate_ptrs(ptrs, sizes, allocator);
    randomFillDevicePtrs(ptrs, sizes);

    Kernels::BatchMatmul::forward_kernel(stream,
                                         handle,
                                         output,
                                         a_input,
                                         b_input,
                                         m,
                                         n,
                                         k,
                                         batch,
                                         a_seq_length_dim,
                                         b_seq_length_dim,
                                         seq_length);

    std::vector<float> host_output(num_elements_output);
    cudaMemcpy(host_output.data(),
               output,
               num_elements_output * sizeof(float),
               cudaMemcpyDeviceToHost);

    float *a_grad, *b_grad, *o_grad;
    std::vector<float **> ptrs_grad = {&a_grad, &b_grad, &o_grad};
    allocate_ptrs(ptrs_grad, sizes, allocator);

    Kernels::BatchMatmul::backward_kernel(stream,
                                          handle,
                                          output,
                                          o_grad,
                                          a_input,
                                          a_grad,
                                          b_input,
                                          b_grad,
                                          m,
                                          n,
                                          k,
                                          batch);

    cudaStreamDestroy(stream);
    cudaFree(handle.workSpace);
  }
}
