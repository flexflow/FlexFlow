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

    TensorShape shape = get_float_tensor_shape({batch_size, feature_size});
    TensorShape feature_shape = get_float_tensor_shape({feature_size});

    PerDeviceFFHandle handle;
    setPerDeviceFFHandle(&handle);
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
        makeReadOnlyAccessor(getRandomFilledAccessorW(shape, allocator));
    GenericTensorAccessorW output_accessor = allocator.allocate_tensor(shape);
    GenericTensorAccessorW gamma_accessor =
        getFilledAccessorW(feature_shape, allocator, 1.0f);
    GenericTensorAccessorW beta_accessor =
        getFilledAccessorW(feature_shape, allocator, 0.0f);

    SUBCASE("Test Layer Norm Forward") {
      Kernels::LayerNorm::forward_kernel(stream,
                                         state,
                                         input_accessor,
                                         output_accessor,
                                         gamma_accessor,
                                         beta_accessor);

      std::vector<float> host_output_data =
          fill_host_data<float>(output_accessor.ptr, num_elements);

      SUBCASE("Test Layer Norm Backward") {
        GenericTensorAccessorR grad_output_accessor =
            makeReadOnlyAccessor(getRandomFilledAccessorW(shape, allocator));
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
            makeReadOnlyAccessor(gamma_accessor),
            gamma_grad_accessor,
            beta_grad_accessor);
      }
    }

    cleanup_test(stream, handle);
  }
}
