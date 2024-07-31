#include "doctest/doctest.h"
#include "kernels/pool_2d_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Pool2D Forward and Backward Kernel") {
    size_t input_w = 10, input_h = 10, input_c = 3, input_n = 1;
    size_t output_w = 5, output_h = 5, output_c = 3, output_n = 1;
    size_t pad_h = 0, pad_w = 0, kernel_h = 2, kernel_w = 2, stride_h = 2,
           stride_w = 2;

    PoolOp pool_type = PoolOp::MAX;

    ManagedPerDeviceFFHandle managed_handle{};
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    Pool2DPerDeviceState state =
        Kernels::Pool2D::init_kernel(managed_handle.raw_handle(),
                                     std::nullopt,
                                     input_w,
                                     input_h,
                                     input_c,
                                     input_n,
                                     output_w,
                                     output_h,
                                     output_c,
                                     output_n,
                                     pad_h,
                                     pad_w,
                                     kernel_h,
                                     kernel_w,
                                     stride_h,
                                     stride_w,
                                     pool_type);

    TensorShape input_shape = make_tensor_shape_from_legion_dims(
        {input_w, input_h, input_c, input_n}, DataType::FLOAT);
    TensorShape output_shape = make_tensor_shape_from_legion_dims(
        {output_w, output_h, output_c, output_n}, DataType::FLOAT);

    GenericTensorAccessorW input_accessor =
        create_random_filled_accessor_w<DataType::FLOAT>(input_shape,
                                                         allocator);
    GenericTensorAccessorW output_accessor =
        create_random_filled_accessor_w<DataType::FLOAT>(output_shape,
                                                         allocator);

    SUBCASE("forward_kernel") {
      Kernels::Pool2D::forward_kernel(managed_stream.raw_stream(),
                                      state,
                                      input_accessor.ptr,
                                      output_accessor.ptr);

      std::vector<float> host_output_data =
          load_accessor_data<DataType::FLOAT>(output_accessor);
      CHECK(contains_non_zero(host_output_data));
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorW output_grad_accessor =
          create_filled_accessor_w<float>(output_shape, allocator, 1.0f);
      GenericTensorAccessorW input_grad_accessor =
          allocator.allocate_tensor(input_shape);

      Kernels::Pool2D::backward_kernel(managed_stream.raw_stream(),
                                       state,
                                       input_accessor.ptr,
                                       input_grad_accessor.ptr,
                                       output_accessor.ptr,
                                       output_grad_accessor.ptr);

      std::vector<float> host_input_grad =
          load_accessor_data<DataType::FLOAT>(input_grad_accessor);
      CHECK(contains_non_zero(host_input_grad));
    }
  }
}
