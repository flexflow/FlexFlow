#include "kernels/accessor_contains_non_zero_value.h"
#include "kernels/attention_kernels_gpu.h"
#include "kernels/create_random_filled_accessor.h"
#include "kernels/local_cuda_allocator.h"
#include "kernels/managed_ff_stream.h"
#include "kernels/managed_per_device_ff_handle.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("Test multi-head attention kernel") {
    positive_int num_samples = 10_p;
    positive_int num_heads = 4_p;
    positive_int qSize = 64_p;
    positive_int kSize = 64_p;
    positive_int vSize = 64_p;
    positive_int qProjSize = 64_p;
    positive_int kProjSize = 64_p;
    positive_int vProjSize = 64_p;
    positive_int oProjSize = 64_p;
    positive_int qoSeqLength = 20_p;
    positive_int kvSeqLength = 20_p;

    ManagedFFStream managed_stream{};
    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);

    Allocator allocator = create_local_cuda_memory_allocator();

    MHAPerDeviceState state = Kernels::MultiHeadAttention::gpu_init_kernel(
        managed_handle.raw_handle(),
        allocator,
        /*num_samples=*/num_samples.int_from_positive_int(),
        /*num_heads=*/num_heads.int_from_positive_int(),
        /*qSize=*/qSize.int_from_positive_int(),
        /*kSize=*/kSize.int_from_positive_int(),
        /*vSize=*/vSize.int_from_positive_int(),
        /*qProjSize=*/qProjSize.int_from_positive_int(),
        /*kProjSize=*/kProjSize.int_from_positive_int(),
        /*vProjSize=*/vProjSize.int_from_positive_int(),
        /*oProjSize=*/oProjSize.int_from_positive_int(),
        /*qoSeqLength=*/qoSeqLength.int_from_positive_int(),
        /*kvSeqLength=*/kvSeqLength.int_from_positive_int(),
        /*add_bias_kv=*/false);

    TensorShape query_shape = TensorShape{
        TensorDims{FFOrdered{qoSeqLength, num_samples, qSize}},
        DataType::FLOAT,
    };
    TensorShape key_shape = TensorShape{
        TensorDims{FFOrdered{kvSeqLength, num_samples, kSize}},
        DataType::FLOAT,
    };
    TensorShape value_shape = TensorShape{
        TensorDims{FFOrdered{kvSeqLength, num_samples, vSize}},
        DataType::FLOAT,
    };
    TensorShape output_shape = TensorShape{
        TensorDims{FFOrdered{qoSeqLength, num_samples, oProjSize}},
        DataType::FLOAT,
    };
    TensorShape weight_shape = TensorShape{
        TensorDims{FFOrdered{positive_int{state.weightSize}}},
        DataType::FLOAT,
    };

    GenericTensorAccessorW query_accessor =
        create_random_filled_accessor_w(query_shape, allocator);
    GenericTensorAccessorW key_accessor =
        create_random_filled_accessor_w(key_shape, allocator);
    GenericTensorAccessorW value_accessor =
        create_random_filled_accessor_w(value_shape, allocator);
    GenericTensorAccessorW weight_accessor =
        create_random_filled_accessor_w(weight_shape, allocator);

    SUBCASE("gpu_forward_kernel") {
      GenericTensorAccessorW output_accessor =
          allocator.allocate_tensor(output_shape);

      Kernels::MultiHeadAttention::gpu_forward_kernel(
          managed_stream.raw_stream(),
          state,
          query_accessor.get_float_ptr(),
          key_accessor.get_float_ptr(),
          value_accessor.get_float_ptr(),
          weight_accessor.get_float_ptr(),
          output_accessor.get_float_ptr());

      CHECK(contains_non_zero(output_accessor));
    }

    SUBCASE("gpu_backward_kernel") {
      GenericTensorAccessorW query_grad_accessor =
          create_random_filled_accessor_w(query_shape, allocator);
      GenericTensorAccessorW key_grad_accessor =
          create_random_filled_accessor_w(key_shape, allocator);
      GenericTensorAccessorW value_grad_accessor =
          create_random_filled_accessor_w(value_shape, allocator);
      GenericTensorAccessorW weight_grad_accessor =
          create_random_filled_accessor_w(weight_shape, allocator);
      GenericTensorAccessorW output_grad_accessor =
          create_random_filled_accessor_w(output_shape, allocator);

      Kernels::MultiHeadAttention::gpu_backward_kernel(
          managed_stream.raw_stream(),
          state,
          query_accessor.get_float_ptr(),
          query_grad_accessor.get_float_ptr(),
          key_accessor.get_float_ptr(),
          key_grad_accessor.get_float_ptr(),
          value_accessor.get_float_ptr(),
          value_grad_accessor.get_float_ptr(),
          weight_accessor.get_float_ptr(),
          weight_grad_accessor.get_float_ptr(),
          output_grad_accessor.get_float_ptr());
    }

    Kernels::MultiHeadAttention::gpu_cleanup_kernel(allocator, state);
  }
}
