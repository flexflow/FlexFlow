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

    TensorShape input_shape = make_float_tensor_shape_from_legion_dims(
        {input_w, input_h, input_c, input_n});
    TensorShape output_shape = make_float_tensor_shape_from_legion_dims(
        {output_w, output_h, output_c, output_n});

    PoolOp pool_type = PoolOp::MAX;

    ffStream_t stream = create_ff_stream();
    PerDeviceFFHandle handle = get_per_device_ff_handle();

    Allocator allocator = get_local_memory_allocator();

    Pool2DPerDeviceState state = Kernels::Pool2D::init_kernel(handle,
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

    SUBCASE("forward_kernel") {
      GenericTensorAccessorW input_data =
          create_random_filled_accessor_w(input_shape, allocator);
      GenericTensorAccessorW output_data =
          allocator.allocate_tensor(output_shape);

      Kernels::Pool2D::forward_kernel(
          stream, state, input_data.ptr, output_data.ptr);

      std::vector<float> host_output_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(output_data));

      SUBCASE("backward_kernel") {
        GenericTensorAccessorW output_grad =
            create_filled_accessor_w(output_shape, allocator, 1.0f);
        GenericTensorAccessorW input_grad =
            allocator.allocate_tensor(input_shape);

        Kernels::Pool2D::backward_kernel(stream,
                                         state,
                                         input_data.ptr,
                                         input_grad.ptr,
                                         output_data.ptr,
                                         output_grad.ptr);

        std::vector<float> host_input_grad_data =
            load_data_to_host_from_device<float>(
                read_only_accessor_from_write_accessor(input_grad));
        CHECK(contains_non_zero(host_input_grad_data));
      }
    }

    cleanup_test(stream, handle);
  }
}
