#include "doctest/doctest.h"
#include "kernels/gather_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Gather Forward and Backward Kernel") {
    size_t num_elements = 100;
    size_t output_size = 50;

    TensorShape input_shape = get_float_tensor_shape({num_elements});
    TensorShape output_shape = get_float_tensor_shape({output_size});

    PerDeviceFFHandle handle;
    setPerDeviceFFHandle(&handle);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    Allocator allocator = get_local_memory_allocator();

    SUBCASE("Test Gather Forward") {
      GenericTensorAccessorW device_output_accessor =
          getRandomFilledAccessorW(input_shape, allocator);
      GenericTensorAccessorR device_input_accessor = makeReadOnlyAccessor(
          getRandomFilledAccessorW(input_shape, allocator));
      GenericTensorAccessorR device_indices_accessor = makeReadOnlyAccessor(
          getRandomFilledAccessorW(output_shape, allocator));

      GatherPerDeviceState state = {handle, legion_dim_t(2)};
      Kernels::Gather::forward_kernel(stream,
                                      state,
                                      device_input_accessor,
                                      device_indices_accessor,
                                      device_output_accessor);

      std::vector<float> host_output_data =
          fill_host_data<float>(device_output_accessor.ptr, num_elements);
      REQUIRE(contains_non_zero(host_output_data));

      SUBCASE("Test Gather Backward") {
        GenericTensorAccessorR device_output_grad_accessor =
            makeReadOnlyAccessor(
                getRandomFilledAccessorW(input_shape, allocator));
        GenericTensorAccessorR device_index_accessor = makeReadOnlyAccessor(
            getRandomFilledAccessorW(output_shape, allocator));
        GenericTensorAccessorW device_input_grad_accessor =
            allocator.allocate_tensor(input_shape);

        Kernels::Gather::backward_kernel(stream,
                                         state,
                                         device_output_grad_accessor,
                                         device_index_accessor,
                                         device_input_grad_accessor);

        std::vector<float> host_input_grad_data =
            fill_host_data<float>(device_input_grad_accessor.ptr, num_elements);
        REQUIRE(contains_non_zero(host_input_grad_data));
      }
    }

    cleanup_test(stream, handle);
  }
}
