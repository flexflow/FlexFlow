#include "doctest/doctest.h"
#include "kernels/reverse_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Call Reverse Forward and Backward Kernels") {
    std::size_t reverse_dim_size = 10;
    std::size_t in_blk_size = 10;
    std::size_t num_out_blks = 1;

    TensorShape input_shape = make_float_tensor_shape_from_legion_dims({100});
    TensorShape output_shape = input_shape;

    ManagedPerDeviceFFHandle managed_handle{};
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    SUBCASE("forward_kernel") {
      GenericTensorAccessorR input_accessor =
          read_only_accessor_from_write_accessor(
              create_filled_accessor_w(input_shape, allocator, 1.0f));
      GenericTensorAccessorW output_accessor =
          allocator.allocate_tensor(output_shape);

      Kernels::Reverse::forward_kernel(managed_stream.raw_stream(),
                                       input_accessor.get_float_ptr(),
                                       output_accessor.get_float_ptr(),
                                       num_out_blks,
                                       reverse_dim_size,
                                       in_blk_size,
                                       input_accessor.shape.num_elements());

      std::vector<float> check_output_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(output_accessor));
      CHECK(contains_non_zero(check_output_data));
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorW output_grad_accessor =
          create_random_filled_accessor_w(output_shape, allocator);
      GenericTensorAccessorW input_grad_accessor =
          create_random_filled_accessor_w(input_shape, allocator);

      Kernels::Reverse::backward_kernel(
          managed_stream.raw_stream(),
          output_grad_accessor.get_float_ptr(),
          input_grad_accessor.get_float_ptr(),
          num_out_blks,
          reverse_dim_size,
          in_blk_size,
          input_grad_accessor.shape.num_elements());

      std::vector<float> host_grad_input_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(input_grad_accessor));
      CHECK(contains_non_zero(host_grad_input_data));
    }
  }
}
