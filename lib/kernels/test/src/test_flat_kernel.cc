#include "doctest/doctest.h"
#include "kernels/flat_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Flat Kernel") {
    TensorShape input_shape = make_float_tensor_shape_from_legion_dims({100});

    Allocator allocator = get_local_memory_allocator();

    ManagedStream mStream = get_managed_stream();

    GenericTensorAccessorR input_accessor =
        read_only_accessor_from_write_accessor(
            create_filled_accessor_w(input_shape, allocator, 2.0f));
    GenericTensorAccessorW output_accessor =
        create_filled_accessor_w(input_shape, allocator, 2.0f);

    SUBCASE("forward_kernel") {
      Kernels::Flat::forward_kernel(
          mStream.stream, input_accessor, output_accessor.get_float_ptr());

      std::vector<float> check_output_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(output_accessor));

      std::vector<float> expected_output_data(
          input_accessor.shape.num_elements(), 2.0f);
      CHECK(check_output_data == expected_output_data);
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorR data_accessor =
          read_only_accessor_from_write_accessor(
              create_filled_accessor_w(input_shape, allocator, 1.0f));

      Kernels::Flat::backward_kernel(mStream.stream,
                                     input_accessor,
                                     output_accessor.get_float_ptr(),
                                     data_accessor.get_float_ptr());

      std::vector<float> backward_output_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(output_accessor));

      std::vector<float> expected_output_data(
          input_accessor.shape.num_elements(), 3.0f);
      CHECK(backward_output_data == expected_output_data);
    }
  }
}
