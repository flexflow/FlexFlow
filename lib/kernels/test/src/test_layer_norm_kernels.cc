#include "doctest/doctest.h"
#include "kernels/layer_norm_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test LayerNorm Forward and Backward Kernel") {
    size_t batch_size = 10;
    size_t feature_size = 10;
    float epsilon = 1e-5f;
    bool elementwise_affine = true;

    TensorShape input_shape = make_tensor_shape_from_legion_dims(
        {batch_size, feature_size}, DataType::FLOAT);
    TensorShape output_shape = input_shape;
    TensorShape feature_shape =
        make_tensor_shape_from_legion_dims({feature_size}, DataType::FLOAT);

    ManagedPerDeviceFFHandle managed_handle{};
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    LayerNormPerDeviceState state =
        Kernels::LayerNorm::init_kernel(managed_handle.raw_handle(),
                                        allocator,
                                        elementwise_affine,
                                        batch_size,
                                        feature_size,
                                        epsilon);

    GenericTensorAccessorR input_accessor =
        create_random_filled_accessor_r<DataType::FLOAT>(input_shape,
                                                         allocator);
    GenericTensorAccessorW gamma_accessor =
        create_filled_accessor_w<float>(feature_shape, allocator, 1.0f);

    SUBCASE("forward_kernel") {
      GenericTensorAccessorW output_accessor =
          allocator.allocate_tensor(output_shape);
      GenericTensorAccessorW beta_accessor =
          create_filled_accessor_w<float>(feature_shape, allocator, 0.0f);

      Kernels::LayerNorm::forward_kernel(managed_stream.raw_stream(),
                                         state,
                                         input_accessor,
                                         output_accessor,
                                         gamma_accessor,
                                         beta_accessor);
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorR output_grad_accessor =
          create_random_filled_accessor_r<DataType::FLOAT>(output_shape,
                                                           allocator);
      GenericTensorAccessorW input_grad_accessor =
          create_random_filled_accessor_w<DataType::FLOAT>(input_shape,
                                                           allocator);
      GenericTensorAccessorW gamma_grad_accessor =
          allocator.allocate_tensor(feature_shape);
      GenericTensorAccessorW beta_grad_accessor =
          allocator.allocate_tensor(feature_shape);

      Kernels::LayerNorm::backward_kernel(
          managed_stream.raw_stream(),
          state,
          output_grad_accessor,
          input_accessor,
          input_grad_accessor,
          read_only_accessor_from_write_accessor(gamma_accessor),
          gamma_grad_accessor,
          beta_grad_accessor);
    }
  }
}
