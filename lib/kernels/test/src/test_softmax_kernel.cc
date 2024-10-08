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
        create_random_filled_accessor_w<DataType::FLOAT>(output_shape,
                                                         allocator);

    SUBCASE("forward_kernel") {
      GenericTensorAccessorW input_accessor =
          create_random_filled_accessor_w<DataType::FLOAT>(input_shape,
                                                           allocator);

      Kernels::Softmax::forward_kernel(managed_stream.raw_stream(),
                                       state,
                                       input_accessor.get_float_ptr(),
                                       output_accessor.get_float_ptr());

      std::vector<float> host_output_data =
          load_accessor_data<DataType::FLOAT>(output_accessor);
      CHECK(contains_non_zero(host_output_data));
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorW output_grad_accessor =
          create_filled_accessor_w<float>(output_shape, allocator, 1.0f);
      GenericTensorAccessorW input_grad_accessor =
          allocator.allocate_tensor(input_shape);

      Kernels::Softmax::backward_kernel(
          managed_stream.raw_stream(),
          input_grad_accessor.get_float_ptr(),
          output_grad_accessor.get_float_ptr(),
          output_grad_accessor.shape.num_elements());

      std::vector<float> expected_input_grad_data =
          std::vector<float>(input_grad_accessor.shape.num_elements(), 1.0f);
      std::vector<float> host_input_grad_data =
          load_accessor_data<DataType::FLOAT>(input_grad_accessor);
      CHECK(vectors_are_approx_equal(host_input_grad_data,
                                     expected_input_grad_data));
    }
  }
}
