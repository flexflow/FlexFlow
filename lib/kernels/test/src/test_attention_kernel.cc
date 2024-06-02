#include "doctest/doctest.h"
#include "kernels/attention_kernels.h"
#include "kernels/local_allocator.h"
#include "test_utils.h"

void allocate_ptrs(std::vector<void **> &gpu_data_ptrs,
                   const std::vector<size_t> &num_elements,
                   Allocator &allocator) {
  for (size_t i = 0; i < gpu_data_ptrs.size(); ++i) {
    *gpu_data_ptrs[i] = allocator.allocate(num_elements[i] * sizeof(float));
  }
}

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test multi-head attention forward kernel") {
    int num_samples = 10;
    int num_heads = 4;
    int qSize = 64, kSize = 64, vSize = 64;
    int qProjSize = 64, kProjSize = 64, vProjSize = 64, oProjSize = 64;
    int qoSeqLength = 20, kvSeqLength = 20;

    size_t query_size = num_samples * qoSeqLength * qSize;
    size_t key_size = num_samples * kvSeqLength * kSize;
    size_t value_size = num_samples * kvSeqLength * vSize;
    size_t output_size = num_samples * qoSeqLength * oProjSize;

    Allocator allocator = get_local_memory_allocator();

    PerDeviceFFHandle handle;
    setPerDeviceFFHandle(&handle);

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    MHAPerDeviceState state = Kernels::MultiHeadAttention::init_kernel(
        handle, allocator, num_samples, num_heads, qSize, kSize, vSize,
        qProjSize, kProjSize, vProjSize, oProjSize, qoSeqLength, kvSeqLength,
        false);

    void *query_ptr, *key_ptr, *value_ptr, *weight_ptr, *output_ptr;

    std::vector<void**> ptrs = {&query_ptr, &key_ptr, &value_ptr, &weight_ptr,
                                 &output_ptr};
    std::vector<size_t> sizes = {query_size, key_size, value_size,
                                 state.weightSize, output_size};

    allocate_ptrs(ptrs, sizes, allocator);
    randomFillDevicePtrs(ptrs, sizes);

    Kernels::MultiHeadAttention::forward_kernel(
        stream, state, static_cast<float *>(query_ptr),
        static_cast<float *>(key_ptr), static_cast<float *>(value_ptr),
        static_cast<float *>(weight_ptr), static_cast<float *>(output_ptr));

    std::vector<float> host_output(num_samples * qoSeqLength * oProjSize);
    checkCUDA(cudaMemcpy(host_output.data(), output_ptr,
                         host_output.size() * sizeof(float),
                         cudaMemcpyDeviceToHost));

    REQUIRE(contains_non_zero(host_output));

    checkCUDA(cudaStreamDestroy(stream));
    Kernels::MultiHeadAttention::cleanup_kernel(allocator, state);
  }

  TEST_CASE("Test multi-head attention backward kernel") {
    int num_samples = 10;
    int num_heads = 4;
    int qSize = 64, kSize = 64, vSize = 64;
    int qProjSize = 64, kProjSize = 64, vProjSize = 64, oProjSize = 64;
    int qoSeqLength = 20, kvSeqLength = 20;

    size_t query_size = num_samples * qoSeqLength * qSize;
    size_t key_size = num_samples * kvSeqLength * kSize;
    size_t value_size = num_samples * kvSeqLength * vSize;
    size_t output_size = num_samples * qoSeqLength * oProjSize;

    Allocator allocator = get_local_memory_allocator();

    PerDeviceFFHandle handle;
    setPerDeviceFFHandle(&handle);
    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    MHAPerDeviceState state = Kernels::MultiHeadAttention::init_kernel(
        handle, allocator, num_samples, num_heads, qSize, kSize, vSize,
        qProjSize, kProjSize, vProjSize, oProjSize, qoSeqLength, kvSeqLength,
        false);

    void *query_ptr, *key_ptr, *value_ptr, *weight_ptr, *output_ptr;
    void *query_grad_ptr, *key_grad_ptr, *value_grad_ptr, *weight_grad_ptr,
        *output_grad_ptr;

    std::vector<void**> ptrs = {&query_ptr, &key_ptr, &value_ptr, &weight_ptr,
                                 &output_ptr};
    std::vector<void**> grad_ptrs = {&query_grad_ptr, &key_grad_ptr,
                                      &value_grad_ptr, &weight_grad_ptr,
                                      &output_grad_ptr};

    std::vector<size_t> sizes = {query_size,       key_size,    value_size,
                                 state.weightSize, output_size, output_size};

    allocate_ptrs(ptrs, sizes, allocator);
    allocate_ptrs(grad_ptrs, sizes, allocator);
    randomFillDevicePtrs(ptrs, sizes);
    randomFillDevicePtrs(grad_ptrs, sizes);

    Kernels::MultiHeadAttention::backward_kernel(
        stream, state, static_cast<float *>(query_ptr),
        static_cast<float *>(query_grad_ptr), static_cast<float *>(key_ptr),
        static_cast<float *>(key_grad_ptr), static_cast<float *>(value_ptr),
        static_cast<float *>(value_grad_ptr), static_cast<float *>(weight_ptr),
        static_cast<float *>(weight_grad_ptr),
        static_cast<float *>(output_grad_ptr));

    std::vector<float> output_grad(num_samples * qoSeqLength * oProjSize);

    checkCUDA(cudaMemcpy(output_grad.data(), output_grad_ptr,
                         output_grad.size() * sizeof(float),
                         cudaMemcpyDeviceToHost));

    REQUIRE(contains_non_zero(output_grad));

    checkCUDA(cudaStreamDestroy(stream));

    Kernels::MultiHeadAttention::cleanup_kernel(allocator, state);
  }
}