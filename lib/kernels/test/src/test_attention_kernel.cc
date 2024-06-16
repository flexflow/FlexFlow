#include "doctest/doctest.h"
#include "kernels/attention_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test multi-head attention kernel") {
    size_t num_samples = 10;
    size_t num_heads = 4;
    size_t qSize = 64, kSize = 64, vSize = 64;
    size_t qProjSize = 64, kProjSize = 64, vProjSize = 64, oProjSize = 64;
    size_t qoSeqLength = 20, kvSeqLength = 20;

    PerDeviceFFHandle handle;
    setPerDeviceFFHandle(&handle);

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Allocator allocator = get_local_memory_allocator();

    MHAPerDeviceState state =
        Kernels::MultiHeadAttention::init_kernel(handle,
                                                 allocator,
                                                 num_samples,
                                                 num_heads,
                                                 qSize,
                                                 kSize,
                                                 vSize,
                                                 qProjSize,
                                                 kProjSize,
                                                 vProjSize,
                                                 oProjSize,
                                                 qoSeqLength,
                                                 kvSeqLength,
                                                 false);

    TensorShape query_shape =
        get_float_tensor_shape({qoSeqLength, num_samples, qSize});
    TensorShape key_shape =
        get_float_tensor_shape({kvSeqLength, num_samples, kSize});
    TensorShape value_shape =
        get_float_tensor_shape({kvSeqLength, num_samples, vSize});
    TensorShape output_shape =
        get_float_tensor_shape({qoSeqLength, num_samples, oProjSize});
    TensorShape weight_shape = get_float_tensor_shape({state.weightSize});

    SUBCASE("Test multi-head attention forward kernel") {
      GenericTensorAccessorW query_accessor =
          getRandomFilledAccessorW(query_shape, allocator);
      GenericTensorAccessorW key_accessor =
          getRandomFilledAccessorW(key_shape, allocator);
      GenericTensorAccessorW value_accessor =
          getRandomFilledAccessorW(value_shape, allocator);
      GenericTensorAccessorW weight_accessor =
          getRandomFilledAccessorW(weight_shape, allocator);
      GenericTensorAccessorW output_accessor =
          allocator.allocate_tensor(output_shape);

      Kernels::MultiHeadAttention::forward_kernel(stream,
                                                  state,
                                                  (float *)query_accessor.ptr,
                                                  (float *)key_accessor.ptr,
                                                  (float *)value_accessor.ptr,
                                                  (float *)weight_accessor.ptr,
                                                  (float *)output_accessor.ptr);

      std::vector<float> host_output = fill_host_data<float>(
          output_accessor.ptr, num_samples * qoSeqLength * oProjSize);
      REQUIRE(contains_non_zero(host_output));

      SUBCASE("Test multi-head attention backward kernel") {
        GenericTensorAccessorW query_grad_accessor =
            getRandomFilledAccessorW(query_shape, allocator);
        GenericTensorAccessorW key_grad_accessor =
            getRandomFilledAccessorW(key_shape, allocator);
        GenericTensorAccessorW value_grad_accessor =
            getRandomFilledAccessorW(value_shape, allocator);
        GenericTensorAccessorW weight_grad_accessor =
            getRandomFilledAccessorW(weight_shape, allocator);
        GenericTensorAccessorW output_grad_accessor =
            getRandomFilledAccessorW(output_shape, allocator);

        Kernels::MultiHeadAttention::backward_kernel(
            stream,
            state,
            (float *)query_accessor.ptr,
            (float *)query_grad_accessor.ptr,
            (float *)key_accessor.ptr,
            (float *)key_grad_accessor.ptr,
            (float *)value_accessor.ptr,
            (float *)value_grad_accessor.ptr,
            (float *)weight_accessor.ptr,
            (float *)weight_grad_accessor.ptr,
            (float *)output_grad_accessor.ptr);

        std::vector<float> output_grad = fill_host_data<float>(
            output_grad_accessor.ptr, num_samples * qoSeqLength * oProjSize);

        REQUIRE(contains_non_zero(output_grad));
      }
    }

    cleanup_test(stream, handle);
    Kernels::MultiHeadAttention::cleanup_kernel(allocator, state);
  }
}
