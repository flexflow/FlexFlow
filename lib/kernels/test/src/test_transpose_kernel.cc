#include "doctest/doctest.h"
#include "kernels/transpose_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Transpose Kernel Operations") {
    std::size_t num_dims = 2;
    TensorShape shape = make_float_tensor_shape_from_legion_dims({10, 10});

    std::vector<ff_dim_t> perm = {ff_dim_t(0), ff_dim_t(1)};

    PerDeviceFFHandle handle = get_per_device_ff_handle();
    ffStream_t stream = create_ff_stream();

    Allocator allocator = get_local_memory_allocator();

    TransposePerDeviceState state =
        Kernels::Transpose::init_kernel(num_dims, perm);

    SUBCASE("forward_kernel") {
      GenericTensorAccessorR input_accessor =
          read_only_accessor_from_write_accessor(
              create_random_filled_accessor_w(shape, allocator));
      GenericTensorAccessorW output_accessor = allocator.allocate_tensor(shape);

      Kernels::Transpose::forward_kernel(
          stream, state, input_accessor, output_accessor);

      std::vector<float> host_output_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(output_accessor));

      SUBCASE("backward_kernel") {
        GenericTensorAccessorW input_grad_accessor =
            create_random_filled_accessor_w(shape, allocator);

        Kernels::Transpose::backward_kernel(
            stream,
            state,
            input_grad_accessor,
            read_only_accessor_from_write_accessor(output_accessor));

        std::vector<float> host_grad_input_data =
            load_data_to_host_from_device<float>(
                read_only_accessor_from_write_accessor(input_grad_accessor));
        CHECK(contains_non_zero(host_grad_input_data));
      }
    }

    cleanup_test(stream, handle);
  }
}
