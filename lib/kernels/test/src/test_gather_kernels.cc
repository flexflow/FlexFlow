#include "doctest/doctest.h"
#include "kernels/gather_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Gather Forward and Backward Kernel") {
    TensorShape input_shape = make_float_tensor_shape_from_legion_dims({100});
    TensorShape output_shape = make_float_tensor_shape_from_legion_dims({50});

    ManagedStream mStream = get_managed_stream();
    ManagedHandle mHandle = get_managed_handle();

    Allocator allocator = get_local_memory_allocator();

    GatherPerDeviceState state = {mHandle.handle, legion_dim_t(2)};

    GenericTensorAccessorW device_output_accessor =
        create_random_filled_accessor_w(input_shape, allocator);
    GenericTensorAccessorR device_indices_accessor =
        read_only_accessor_from_write_accessor(
            create_random_filled_accessor_w(output_shape, allocator));

    SUBCASE("forward_kernel") {
      GenericTensorAccessorR device_input_accessor =
          read_only_accessor_from_write_accessor(
              create_random_filled_accessor_w(input_shape, allocator));

      Kernels::Gather::forward_kernel(mStream.stream,
                                      state,
                                      device_input_accessor,
                                      device_indices_accessor,
                                      device_output_accessor);

      std::vector<float> host_output_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(device_output_accessor));
      CHECK(contains_non_zero(host_output_data));
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorW device_input_grad_accessor =
          allocator.allocate_tensor(input_shape);

      Kernels::Gather::backward_kernel(
          mStream.stream,
          state,
          read_only_accessor_from_write_accessor(device_output_accessor),
          device_indices_accessor,
          device_input_grad_accessor);

      std::vector<float> host_input_grad_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(
                  device_input_grad_accessor));
      CHECK(contains_non_zero(host_input_grad_data));
    }
  }
}
