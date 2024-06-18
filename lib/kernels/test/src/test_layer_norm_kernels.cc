#include "doctest/doctest.h"
#include "kernels/layer_norm_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test LayerNorm Forward and Backward Kernel") {
    size_t batch_size = 10;
    size_t feature_size = 10;
    size_t num_elements = batch_size * feature_size;
    float epsilon = 1e-5f;
    bool elementwise_affine = true;

    TensorShape shape =
        make_float_tensor_shape_w_legion_dims({batch_size, feature_size});
    TensorShape feature_shape =
        make_float_tensor_shape_w_legion_dims({feature_size});

    PerDeviceFFHandle handle = get_per_device_ff_handle();
    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

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
    GenericTensorAccessorW output_accessor = allocator.allocate_tensor(shape);
    GenericTensorAccessorW gamma_accessor =
        create_filled_accessor_w(feature_shape, allocator, 1.0f);
    GenericTensorAccessorW beta_accessor =
        create_filled_accessor_w(feature_shape, allocator, 0.0f);

    SUBCASE("forward_kernel") {
      Kernels::LayerNorm::forward_kernel(stream,
                                         state,
                                         input_accessor,
                                         output_accessor,
                                         gamma_accessor,
                                         beta_accessor);

      std::vector<float> host_output_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(output_accessor));

      SUBCASE("backward_kernel") {
        GenericTensorAccessorR grad_output_accessor =
            read_only_accessor_from_write_accessor(
                create_random_filled_accessor_w(shape, allocator));
        GenericTensorAccessorW grad_input_accessor =
            allocator.allocate_tensor(shape);
        GenericTensorAccessorW gamma_grad_accessor =
            allocator.allocate_tensor(feature_shape);
        GenericTensorAccessorW beta_grad_accessor =
            allocator.allocate_tensor(feature_shape);

        Kernels::LayerNorm::backward_kernel(
            stream,
            state,
            grad_output_accessor,
            input_accessor,
            grad_input_accessor,
            read_only_accessor_from_write_accessor(gamma_accessor),
            gamma_grad_accessor,
            beta_grad_accessor);
      }
    }

    cleanup_test(stream, handle);
  }
}
