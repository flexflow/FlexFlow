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

    ManagedFFStream managed_stream{};
    ManagedPerDeviceFFHandle managed_handle{};

    Allocator allocator = create_local_cuda_memory_allocator();

    MHAPerDeviceState state =
        Kernels::MultiHeadAttention::init_kernel(managed_handle.raw_handle(),
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

    TensorShape query_shape = make_tensor_shape_from_legion_dims(
        {qoSeqLength, num_samples, qSize}, DataType::FLOAT);
    TensorShape key_shape = make_tensor_shape_from_legion_dims(
        {kvSeqLength, num_samples, kSize}, DataType::FLOAT);
    TensorShape value_shape = make_tensor_shape_from_legion_dims(
        {kvSeqLength, num_samples, vSize}, DataType::FLOAT);
    TensorShape output_shape = make_tensor_shape_from_legion_dims(
        {qoSeqLength, num_samples, oProjSize}, DataType::FLOAT);
    TensorShape weight_shape =
        make_tensor_shape_from_legion_dims({state.weightSize}, DataType::FLOAT);

    GenericTensorAccessorW query_accessor =
        create_random_filled_accessor_w<DataType::FLOAT>(query_shape,
                                                         allocator);
    GenericTensorAccessorW key_accessor =
        create_random_filled_accessor_w<DataType::FLOAT>(key_shape, allocator);
    GenericTensorAccessorW value_accessor =
        create_random_filled_accessor_w<DataType::FLOAT>(value_shape,
                                                         allocator);
    GenericTensorAccessorW weight_accessor =
        create_random_filled_accessor_w<DataType::FLOAT>(weight_shape,
                                                         allocator);

    SUBCASE("forward_kernel") {
      GenericTensorAccessorW output_accessor =
          allocator.allocate_tensor(output_shape);

      Kernels::MultiHeadAttention::forward_kernel(
          managed_stream.raw_stream(),
          state,
          query_accessor.get_float_ptr(),
          key_accessor.get_float_ptr(),
          value_accessor.get_float_ptr(),
          weight_accessor.get_float_ptr(),
          output_accessor.get_float_ptr());

      std::vector<float> host_output =
          load_accessor_data<DataType::FLOAT>(output_accessor);
      CHECK(contains_non_zero(host_output));
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorW query_grad_accessor =
          create_random_filled_accessor_w<DataType::FLOAT>(query_shape,
                                                           allocator);
      GenericTensorAccessorW key_grad_accessor =
          create_random_filled_accessor_w<DataType::FLOAT>(key_shape,
                                                           allocator);
      GenericTensorAccessorW value_grad_accessor =
          create_random_filled_accessor_w<DataType::FLOAT>(value_shape,
                                                           allocator);
      GenericTensorAccessorW weight_grad_accessor =
          create_random_filled_accessor_w<DataType::FLOAT>(weight_shape,
                                                           allocator);
      GenericTensorAccessorW output_grad_accessor =
          create_random_filled_accessor_w<DataType::FLOAT>(output_shape,
                                                           allocator);

      Kernels::MultiHeadAttention::backward_kernel(
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

    Kernels::MultiHeadAttention::cleanup_kernel(allocator, state);
  }
}
