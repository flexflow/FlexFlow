#include "doctest/doctest.h"
#include "kernels/reduction_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Reduction Forward and Backward Kernel") {
    std::size_t num_replicas = 5;

    TensorShape expanded_shape =
        make_float_tensor_shape_from_legion_dims({10, 10, 10, 10, 10});
    TensorShape shape = make_float_tensor_shape_from_legion_dims({10});

    ManagedStream mStream = get_managed_stream();

    Allocator allocator = get_local_memory_allocator();

    GenericTensorAccessorW output_accessor =
        create_random_filled_accessor_w(shape, allocator);

    SUBCASE("forward_kernel") {
      GenericTensorAccessorR input_accessor =
          read_only_accessor_from_write_accessor(
              create_random_filled_accessor_w(expanded_shape, allocator));

      Kernels::Reduction::forward_kernel(
          mStream.stream, input_accessor, output_accessor, num_replicas);

      std::vector<float> host_output_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(output_accessor));
      CHECK(contains_non_zero(host_output_data));
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorR grad_accessor =
          read_only_accessor_from_write_accessor(
              create_filled_accessor_w(shape, allocator, 1.0f));

      Kernels::Reduction::backward_kernel(
          mStream.stream, output_accessor, grad_accessor);

      std::vector<float> host_grad_data = load_data_to_host_from_device<float>(
          read_only_accessor_from_write_accessor(output_accessor));
      CHECK(contains_non_zero(host_grad_data));
    }
  }
}
