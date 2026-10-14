#include "kernels/create_constant_filled_accessor_r.h"
#include "kernels/create_random_filled_accessor.h"
#include "kernels/layer_norm_kernels_gpu.h"
#include "kernels/local_cuda_allocator.h"
#include "kernels/managed_ff_stream.h"
#include "kernels/managed_per_device_ff_handle.h"
#include "op-attrs/datatype_value.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("Test LayerNorm Forward and Backward Kernel") {
    positive_int batch_size = 10_p;
    positive_int feature_size = 10_p;
    float epsilon = 1e-5f;
    bool elementwise_affine = true;

    TensorShape input_shape = TensorShape{
        TensorDims{FFOrdered{batch_size, feature_size}},
        DataType::FLOAT,
    };
    TensorShape output_shape = input_shape;
    TensorShape feature_shape = TensorShape{
        TensorDims{FFOrdered{feature_size}},
        DataType::FLOAT,
    };

    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    LayerNormPerDeviceState state = Kernels::LayerNorm::gpu_init_kernel(
        managed_handle.raw_handle(),
        allocator,
        elementwise_affine,
        batch_size.int_from_positive_int(),
        feature_size.int_from_positive_int(),
        epsilon);

    GenericTensorAccessorR input_accessor =
        create_random_filled_accessor_r(input_shape, allocator);
    GenericTensorAccessorW gamma_accessor = create_constant_filled_accessor_w(
        feature_shape, allocator, make_float_data_type_value(1));

    SUBCASE("gpu_forward_kernel") {
      GenericTensorAccessorW output_accessor =
          allocator.allocate_tensor(output_shape);
      GenericTensorAccessorW beta_accessor = create_constant_filled_accessor_w(
          feature_shape, allocator, make_float_data_type_value(0));

      Kernels::LayerNorm::gpu_forward_kernel(managed_stream.raw_stream(),
                                             state,
                                             input_accessor,
                                             output_accessor,
                                             gamma_accessor,
                                             beta_accessor);
    }

    SUBCASE("gpu_backward_kernel") {
      GenericTensorAccessorR output_grad_accessor =
          create_random_filled_accessor_r(output_shape, allocator);
      GenericTensorAccessorW input_grad_accessor =
          create_random_filled_accessor_w(input_shape, allocator);
      GenericTensorAccessorW gamma_grad_accessor =
          allocator.allocate_tensor(feature_shape);
      GenericTensorAccessorW beta_grad_accessor =
          allocator.allocate_tensor(feature_shape);

      Kernels::LayerNorm::gpu_backward_kernel(
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
