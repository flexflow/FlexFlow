#include "doctest/doctest.h"
#include "kernels/attention_kernels.h"
#include "kernels/local_allocator.h"
#include <iostream>

#include <random>

namespace FlexFlow {
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test multi-head attention forward kernel") {
    int num_samples = 10;
    int num_heads = 4;
    int qSize = 64, kSize = 64, vSize = 64;
    int qProjSize = 64, kProjSize = 64, vProjSize = 64, oProjSize = 64;
    int qoSeqLength = 20, kvSeqLength = 20;

    Allocator allocator = get_local_memory_allocator();

    PerDeviceFFHandle handle;
    cudnnCreate(&handle.dnn);
    cublasCreate(&handle.blas);
    handle.workSpaceSize = 1024 * 1024;
    cudaMalloc(&handle.workSpace, handle.workSpaceSize);
    handle.allowTensorOpMathConversion = true;

    MHAPerDeviceState state = Kernels::MultiHeadAttention::init_kernel(
        handle, allocator, num_samples, num_heads, qSize, kSize, vSize,
        qProjSize, kProjSize, vProjSize, oProjSize, qoSeqLength, kvSeqLength,
        false);

    void *query_ptr =
        allocator.allocate(num_samples * qoSeqLength * qSize * sizeof(float));
    void *key_ptr =
        allocator.allocate(num_samples * kvSeqLength * kSize * sizeof(float));
    void *value_ptr =
        allocator.allocate(num_samples * kvSeqLength * vSize * sizeof(float));
    void *weight_ptr = allocator.allocate(state.weightSize);
    void *output_ptr = allocator.allocate(num_samples * qoSeqLength *
                                          oProjSize * sizeof(float));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> host_query(num_samples * qoSeqLength * qSize);
    std::vector<float> host_key(num_samples * kvSeqLength * kSize);
    std::vector<float> host_value(num_samples * kvSeqLength * vSize);
    std::vector<float> host_weight(state.weightSize / sizeof(float));

    for (auto &val : host_query)
      val = dist(gen);
    for (auto &val : host_key)
      val = dist(gen);
    for (auto &val : host_value)
      val = dist(gen);
    for (auto &val : host_weight)
      val = dist(gen);

    checkCUDA(cudaMemcpy(query_ptr, host_query.data(),
                         host_query.size() * sizeof(float),
                         cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpy(key_ptr, host_key.data(),
                         host_key.size() * sizeof(float),
                         cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpy(value_ptr, host_value.data(),
                         host_value.size() * sizeof(float),
                         cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpy(weight_ptr, host_weight.data(),
                         host_weight.size() * sizeof(float),
                         cudaMemcpyHostToDevice));

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Kernels::MultiHeadAttention::forward_kernel(
        stream, state, static_cast<float *>(query_ptr),
        static_cast<float *>(key_ptr), static_cast<float *>(value_ptr),
        static_cast<float *>(weight_ptr), static_cast<float *>(output_ptr));

    std::vector<float> host_output(num_samples * qoSeqLength * oProjSize);
    checkCUDA(cudaMemcpy(host_output.data(), output_ptr,
                         host_output.size() * sizeof(float),
                         cudaMemcpyDeviceToHost));

    // TODO: PROBABLY NEED DIFFERENT CHECK?!!??!
    REQUIRE(std::any_of(host_output.begin(), host_output.end(),
                        [](float v) { return v != 0; }));

    checkCUDA(cudaStreamDestroy(stream));
    Kernels::MultiHeadAttention::cleanup_kernel(allocator, state);
  }

  TEST_CASE("Test multi-head attention backward kernel") {
    int num_samples = 10;
    int num_heads = 4;
    int qSize = 64, kSize = 64, vSize = 64;
    int qProjSize = 64, kProjSize = 64, vProjSize = 64, oProjSize = 64;
    int qoSeqLength = 20, kvSeqLength = 20;

    Allocator allocator = get_local_memory_allocator();

    PerDeviceFFHandle handle;
    cudnnCreate(&handle.dnn);
    cublasCreate(&handle.blas);
    handle.workSpaceSize = 1024 * 1024;
    cudaMalloc(&handle.workSpace, handle.workSpaceSize);
    handle.allowTensorOpMathConversion = true;

    MHAPerDeviceState state = Kernels::MultiHeadAttention::init_kernel(
        handle, allocator, num_samples, num_heads, qSize, kSize, vSize,
        qProjSize, kProjSize, vProjSize, oProjSize, qoSeqLength, kvSeqLength,
        false);

    void *query_ptr =
        allocator.allocate(num_samples * qoSeqLength * qSize * sizeof(float));
    void *key_ptr =
        allocator.allocate(num_samples * kvSeqLength * kSize * sizeof(float));
    void *value_ptr =
        allocator.allocate(num_samples * kvSeqLength * vSize * sizeof(float));
    void *weight_ptr = allocator.allocate(state.weightSize);
    void *output_ptr = allocator.allocate(num_samples * qoSeqLength *
                                          oProjSize * sizeof(float));

    void *query_grad_ptr =
        allocator.allocate(num_samples * qoSeqLength * qSize * sizeof(float));
    void *key_grad_ptr =
        allocator.allocate(num_samples * kvSeqLength * kSize * sizeof(float));
    void *value_grad_ptr =
        allocator.allocate(num_samples * kvSeqLength * vSize * sizeof(float));
    void *weight_grad_ptr = allocator.allocate(state.weightSize);
    void *output_grad_ptr = allocator.allocate(num_samples * qoSeqLength *
                                               oProjSize * sizeof(float));

    cudaMemset(query_grad_ptr, 0,
               num_samples * qoSeqLength * qSize * sizeof(float));
    cudaMemset(key_grad_ptr, 0,
               num_samples * kvSeqLength * kSize * sizeof(float));
    cudaMemset(value_grad_ptr, 0,
               num_samples * kvSeqLength * vSize * sizeof(float));
    cudaMemset(weight_grad_ptr, 0, state.weightSize);
    cudaMemset(output_grad_ptr, 0,
               num_samples * qoSeqLength * oProjSize * sizeof(float));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> host_query(num_samples * qoSeqLength * qSize);
    std::vector<float> host_key(num_samples * kvSeqLength * kSize);
    std::vector<float> host_value(num_samples * kvSeqLength * vSize);
    std::vector<float> host_weight(state.weightSize / sizeof(float));
    std::vector<float> host_output(num_samples * qoSeqLength * oProjSize);
    std::vector<float> host_output_grad(num_samples * qoSeqLength * oProjSize);

    for (auto &val : host_query)
      val = dist(gen);
    for (auto &val : host_key)
      val = dist(gen);
    for (auto &val : host_value)
      val = dist(gen);
    for (auto &val : host_weight)
      val = dist(gen);
    for (auto &val : host_output)
      val = dist(gen);
    for (auto &val : host_output_grad)
      val = dist(gen);

    checkCUDA(cudaMemcpy(query_ptr, host_query.data(),
                         host_query.size() * sizeof(float),
                         cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpy(key_ptr, host_key.data(),
                         host_key.size() * sizeof(float),
                         cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpy(value_ptr, host_value.data(),
                         host_value.size() * sizeof(float),
                         cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpy(weight_ptr, host_weight.data(),
                         host_weight.size() * sizeof(float),
                         cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpy(output_ptr, host_output.data(),
                         host_output.size() * sizeof(float),
                         cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpy(output_grad_ptr, host_output_grad.data(),
                         host_output_grad.size() * sizeof(float),
                         cudaMemcpyHostToDevice));

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

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

    REQUIRE(std::any_of(output_grad.begin(), output_grad.end(),
                        [](float v) { return v != 0; }));

    checkCUDA(cudaStreamDestroy(stream));

    Kernels::MultiHeadAttention::cleanup_kernel(allocator, state);
  }
}
} // namespace FlexFlow
