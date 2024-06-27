#include "doctest/doctest.h"
#include "kernels/replicate_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Replicate Kernel") {
    std::size_t num_replicas = 10;

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

      Kernels::Replicate::forward_kernel(
          managed_stream.raw_stream(), input_accessor, output_accessor);

      std::vector<float> check_output_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(output_accessor));

      std::vector<float> expected_output_data(
          input_accessor.shape.num_elements(), 1.0f);
      CHECK(check_output_data == expected_output_data);
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorW input_grad_accessor =
          create_filled_accessor_w(input_shape, allocator, 1.0f);
      GenericTensorAccessorR output_grad_accessor =
          read_only_accessor_from_write_accessor(
              create_filled_accessor_w(output_shape, allocator, 1.0f));

      Kernels::Replicate::backward_kernel(managed_stream.raw_stream(),
                                          input_grad_accessor,
                                          output_grad_accessor,
                                          num_replicas);

      std::vector<float> check_aggregated_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(input_grad_accessor));
      CHECK(contains_non_zero(check_aggregated_data));
    }
  }
}
