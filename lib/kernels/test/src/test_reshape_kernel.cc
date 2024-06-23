#include "doctest/doctest.h"
#include "kernels/reshape_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Reshape Forward and Backward") {
    TensorShape shape = make_float_tensor_shape_from_legion_dims({100});

    ManagedStream mStream = get_managed_stream();

    Allocator allocator = get_local_memory_allocator();

    ReshapePerDeviceState state =
        Kernels::Reshape::init_kernel(DataType::FLOAT);

    GenericTensorAccessorW output_accessor =
        create_filled_accessor_w(shape, allocator, 1.0f);

    SUBCASE("forward_kernel") {
      GenericTensorAccessorR input_accessor =
          read_only_accessor_from_write_accessor(
              create_filled_accessor_w(shape, allocator, 1.0f));

      Kernels::Reshape::forward_kernel(
          mStream.stream, state, input_accessor, output_accessor);

      std::vector<float> check_output_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(output_accessor));

      std::vector<float> expected_output_data(
          input_accessor.shape.num_elements(), 1.0f);
      CHECK(check_output_data == expected_output_data);
    }

    SUBCASE("backward_kernel") {
      Kernels::Reshape::backward_kernel(
          mStream.stream,
          state,
          output_accessor,
          read_only_accessor_from_write_accessor(output_accessor));

      std::vector<float> host_grad_input_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(output_accessor));

      std::vector<float> expected_grad_input_data(
          output_accessor.shape.num_elements(), 2.0f);
      CHECK(host_grad_input_data == expected_grad_input_data);
    }
  }
}
