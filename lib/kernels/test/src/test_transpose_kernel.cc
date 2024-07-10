#include "doctest/doctest.h"
#include "kernels/transpose_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Transpose Kernel Operations") {
    std::size_t num_dims = 2;

    std::vector<ff_dim_t> perm = {ff_dim_t(0), ff_dim_t(1)};

    ManagedPerDeviceFFHandle managed_handle{};
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    TransposePerDeviceState state =
        Kernels::Transpose::init_kernel(num_dims, perm);

    TensorShape input_shape =
        make_float_tensor_shape_from_legion_dims({10, 10});
    TensorShape output_shape = input_shape;

    SUBCASE("forward_kernel") {
      GenericTensorAccessorR input_accessor =
          read_only_accessor_from_write_accessor(
              create_random_filled_accessor_w(input_shape, allocator));
      GenericTensorAccessorW output_accessor =
          allocator.allocate_tensor(output_shape);

      Kernels::Transpose::forward_kernel(
          managed_stream.raw_stream(), state, input_accessor, output_accessor);

      std::vector<float> host_output_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(output_accessor));
      CHECK(contains_non_zero(host_output_data));
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorR output_grad_accessor =
          read_only_accessor_from_write_accessor(
              create_random_filled_accessor_w(output_shape, allocator));
      GenericTensorAccessorW input_grad_accessor =
          create_random_filled_accessor_w(input_shape, allocator);

      Kernels::Transpose::backward_kernel(managed_stream.raw_stream(),
                                          state,
                                          input_grad_accessor,
                                          output_grad_accessor);

      std::vector<float> host_grad_input_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(input_grad_accessor));
      CHECK(contains_non_zero(host_grad_input_data));
    }
  }
}
