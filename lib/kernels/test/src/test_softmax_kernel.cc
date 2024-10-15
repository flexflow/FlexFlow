#include "doctest/doctest.h"
#include "kernels/softmax_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Softmax Kernel Operations") {
    int input_n = 1, input_c = 1, input_h = 1, input_w = 100, channels = 100;

    ManagedPerDeviceFFHandle managed_handle{};
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    TensorShape input_shape =
        make_tensor_shape_from_legion_dims({100}, DataType::FLOAT);
    TensorShape output_shape = input_shape;

    SoftmaxPerDeviceState state = Kernels::Softmax::init_kernel(
        managed_handle.raw_handle(), 0, input_n, channels, input_h, input_w);

    GenericTensorAccessorW output_accessor =
        create_random_filled_accessor_w(output_shape, allocator);

    SUBCASE("forward_kernel") {
      GenericTensorAccessorW input_accessor =
          create_random_filled_accessor_w(input_shape, allocator);

      Kernels::Softmax::forward_kernel(managed_stream.raw_stream(),
                                       state,
                                       input_accessor.get_float_ptr(),
                                       output_accessor.get_float_ptr());

      CHECK(contains_non_zero(output_accessor));
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorR output_grad_accessor =
          create_random_filled_accessor_r(output_shape, allocator);
      GenericTensorAccessorW input_grad_accessor =
          allocator.allocate_tensor(input_shape);

      Kernels::Softmax::backward_kernel(
          managed_stream.raw_stream(),
          input_grad_accessor.get_float_ptr(),
          output_grad_accessor.get_float_ptr(),
          output_grad_accessor.shape.num_elements());

      CHECK(contains_non_zero(input_grad_accessor));
    }
  }
}
