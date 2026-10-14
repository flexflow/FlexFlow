#include "internal/test_utils.h"
#include "kernels/pool_2d_kernels_gpu.h"
#include "op-attrs/datatype_value.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("Test Pool2D Forward and Backward Kernel") {
    positive_int input_w = 10_p;
    positive_int input_h = 10_p;
    positive_int input_c = 3_p;
    positive_int input_n = 1_p;
    positive_int output_w = 5_p;
    positive_int output_h = 5_p;
    positive_int output_c = 3_p;
    positive_int output_n = 1_p;
    nonnegative_int pad_h = 0_n;
    nonnegative_int pad_w = 0_n;
    positive_int kernel_h = 2_p;
    positive_int kernel_w = 2_p;
    positive_int stride_h = 2_p;
    positive_int stride_w = 2_p;

    PoolOp pool_type = PoolOp::MAX;

    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    Pool2DPerDeviceState state = pool2d_gpu_init_kernel(
        /*handle=*/managed_handle.raw_handle(),
        /*activation=*/std::nullopt,
        /*input_w=*/input_w.int_from_positive_int(),
        /*input_h=*/input_h.int_from_positive_int(),
        /*input_c=*/input_c.int_from_positive_int(),
        /*input_n=*/input_n.int_from_positive_int(),
        /*output_w=*/output_w.int_from_positive_int(),
        /*output_h=*/output_h.int_from_positive_int(),
        /*output_c=*/output_c.int_from_positive_int(),
        /*output_n=*/output_n.int_from_positive_int(),
        /*pad_h=*/pad_h.unwrap_nonnegative(),
        /*pad_w=*/pad_w.unwrap_nonnegative(),
        /*kernel_h=*/kernel_h.int_from_positive_int(),
        /*kernel_w=*/kernel_w.int_from_positive_int(),
        /*stride_h=*/stride_h.int_from_positive_int(),
        /*stride_w=*/stride_w.int_from_positive_int(),
        /*pool_type=*/pool_type);

    TensorShape input_shape = TensorShape{
        TensorDims{FFOrdered{input_n, input_c, input_h, input_w}},
        DataType::FLOAT,
    };
    TensorShape output_shape = TensorShape{
        TensorDims{FFOrdered{output_n, input_c, output_h, output_w}},
        DataType::FLOAT,
    };

    GenericTensorAccessorW input_accessor =
        create_random_filled_accessor_w(input_shape, allocator);
    GenericTensorAccessorW output_accessor =
        create_random_filled_accessor_w(output_shape, allocator);

    SUBCASE("gpu_forward_kernel") {
      pool2d_gpu_forward_kernel(managed_stream.raw_stream(),
                                state,
                                input_accessor.ptr,
                                output_accessor.ptr);

      CHECK(contains_non_zero(output_accessor));
    }

    SUBCASE("gpu_backward_kernel") {
      GenericTensorAccessorW output_grad_accessor = create_filled_accessor_w(
          output_shape, allocator, make_float_data_type_value(1));
      GenericTensorAccessorW input_grad_accessor =
          allocator.allocate_tensor(input_shape);

      pool2d_gpu_backward_kernel(managed_stream.raw_stream(),
                                 state,
                                 output_accessor.ptr,
                                 output_grad_accessor.ptr,
                                 input_accessor.ptr,
                                 input_grad_accessor.ptr);

      CHECK(contains_non_zero(input_grad_accessor));
    }
  }
}
