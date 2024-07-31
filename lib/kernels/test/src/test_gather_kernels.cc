#include "doctest/doctest.h"
#include "kernels/gather_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Gather Forward and Backward Kernel") {
    ManagedPerDeviceFFHandle managed_handle{};
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    GatherPerDeviceState state = {managed_handle.raw_handle(), legion_dim_t(2)};

    TensorShape input_shape = make_float_tensor_shape_from_legion_dims({100});
    TensorShape output_shape = make_float_tensor_shape_from_legion_dims({50});

    GenericTensorAccessorR index_accessor =
        read_only_accessor_from_write_accessor(
            create_random_filled_accessor_w(output_shape, allocator));

    SUBCASE("forward_kernel") {
      GenericTensorAccessorR input_accessor =
          read_only_accessor_from_write_accessor(
              create_random_filled_accessor_w(input_shape, allocator));
      GenericTensorAccessorW output_accessor =
          allocator.allocate_tensor(output_shape);

      Kernels::Gather::forward_kernel(managed_stream.raw_stream(),
                                      state,
                                      input_accessor,
                                      index_accessor,
                                      output_accessor);

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

      Kernels::Gather::backward_kernel(managed_stream.raw_stream(),
                                       state,
                                       output_grad_accessor,
                                       index_accessor,
                                       input_grad_accessor);

      std::vector<float> host_input_grad_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(input_grad_accessor));
      CHECK(contains_non_zero(host_input_grad_data));
    }
  }
}
