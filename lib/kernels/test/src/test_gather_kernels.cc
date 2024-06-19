#include "doctest/doctest.h"
#include "kernels/gather_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Gather Forward and Backward Kernel") {
    TensorShape input_shape = make_float_tensor_shape_from_legion_dims({100});
    TensorShape output_shape = make_float_tensor_shape_from_legion_dims({50});

    ffStream_t stream = create_ff_stream();
    PerDeviceFFHandle handle = get_per_device_ff_handle();

    Allocator allocator = get_local_memory_allocator();

    SUBCASE("forward_kernel") {
      GenericTensorAccessorW device_output_accessor =
          create_random_filled_accessor_w(input_shape, allocator);
      GenericTensorAccessorR device_input_accessor =
          read_only_accessor_from_write_accessor(
              create_random_filled_accessor_w(input_shape, allocator));
      GenericTensorAccessorR device_indices_accessor =
          read_only_accessor_from_write_accessor(
              create_random_filled_accessor_w(output_shape, allocator));

      GatherPerDeviceState state = {handle, legion_dim_t(2)};
      Kernels::Gather::forward_kernel(stream,
                                      state,
                                      device_input_accessor,
                                      device_indices_accessor,
                                      device_output_accessor);

      std::vector<float> host_output_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(device_output_accessor));
      CHECK(contains_non_zero(host_output_data));

      SUBCASE("backward_kernel") {
        GenericTensorAccessorR device_index_accessor =
            read_only_accessor_from_write_accessor(
                create_random_filled_accessor_w(output_shape, allocator));
        GenericTensorAccessorW device_input_grad_accessor =
            allocator.allocate_tensor(input_shape);

        Kernels::Gather::backward_kernel(
            stream,
            state,
            read_only_accessor_from_write_accessor(device_output_accessor),
            device_index_accessor,
            device_input_grad_accessor);

        std::vector<float> host_input_grad_data =
            load_data_to_host_from_device<float>(
                read_only_accessor_from_write_accessor(
                    device_input_grad_accessor));
        CHECK(contains_non_zero(host_input_grad_data));
      }
    }

    cleanup_test(stream, handle);
  }
}
