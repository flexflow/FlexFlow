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

    TensorShape shape =
        make_float_tensor_shape_from_legion_dims({batch_size, feature_size});
    TensorShape feature_shape =
        make_float_tensor_shape_from_legion_dims({feature_size});

    ffStream_t stream = create_ff_stream();
    PerDeviceFFHandle handle = get_per_device_ff_handle();

    Allocator allocator = get_local_memory_allocator();

    LayerNormPerDeviceState state =
        Kernels::LayerNorm::init_kernel(handle,
                                        allocator,
                                        elementwise_affine,
                                        batch_size,
                                        feature_size,
                                        epsilon);

    GenericTensorAccessorR input_accessor =
        read_only_accessor_from_write_accessor(
            create_random_filled_accessor_w(shape, allocator));
    GenericTensorAccessorW gamma_accessor =
        create_filled_accessor_w(feature_shape, allocator, 1.0f);

    SUBCASE("forward_kernel") {
      GenericTensorAccessorW output_accessor = allocator.allocate_tensor(shape);
      GenericTensorAccessorW beta_accessor =
          create_filled_accessor_w(feature_shape, allocator, 0.0f);
      Kernels::LayerNorm::forward_kernel(stream,
                                         state,
                                         input_accessor,
                                         output_accessor,
                                         gamma_accessor,
                                         beta_accessor);

      std::vector<float> host_output_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(output_accessor));
      CHECK(contains_non_zero(host_output_data));
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorR output_accessor =
          read_only_accessor_from_write_accessor(
              create_random_filled_accessor_w(shape, allocator));
      GenericTensorAccessorW grad_input_accessor =
          create_random_filled_accessor_w(shape, allocator);
      GenericTensorAccessorW gamma_grad_accessor =
          allocator.allocate_tensor(feature_shape);
      GenericTensorAccessorW beta_grad_accessor =
          allocator.allocate_tensor(feature_shape);

      Kernels::LayerNorm::backward_kernel(
          stream,
          state,
          output_accessor,
          input_accessor,
          grad_input_accessor,
          read_only_accessor_from_write_accessor(gamma_accessor),
          gamma_grad_accessor,
          beta_grad_accessor);

      std::vector<float> host_grad_input_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(grad_input_accessor));
      std::vector<float> host_gamma_grad_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(gamma_grad_accessor));
      std::vector<float> host_beta_grad_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(beta_grad_accessor));

      CHECK(contains_non_zero(host_grad_input_data));
      CHECK(contains_non_zero(host_gamma_grad_data));
      CHECK(contains_non_zero(host_beta_grad_data));
    }

    cleanup_test(stream, handle);
  }
}
