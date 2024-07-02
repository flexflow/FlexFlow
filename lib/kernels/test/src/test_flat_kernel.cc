#include "doctest/doctest.h"
#include "kernels/flat_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Flat Kernel") {
    Allocator allocator = create_local_cuda_memory_allocator();

    ManagedPerDeviceFFHandle managed_handle{};
    ManagedFFStream managed_stream{};

    TensorShape input_shape = make_float_tensor_shape_from_legion_dims({100});
    TensorShape output_shape = input_shape;

    GenericTensorAccessorR input_accessor =
        read_only_accessor_from_write_accessor(
            create_filled_accessor_w(input_shape, allocator, 2.0f));

    SUBCASE("forward_kernel") {
      GenericTensorAccessorW output_accessor =
          allocator.allocate_tensor(output_shape);

      Kernels::Flat::forward_kernel(managed_stream.raw_stream(),
                                    input_accessor,
                                    output_accessor.get_float_ptr());

      std::vector<float> check_output_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(output_accessor));

      std::vector<float> expected_output_data(
          input_accessor.shape.num_elements(), 2.0f);
      CHECK(check_output_data == expected_output_data);
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorW output_grad_accessor =
          create_filled_accessor_w(output_shape, allocator, 0.0f);
      GenericTensorAccessorW input_grad_accessor =
          create_filled_accessor_w(input_shape, allocator, 1.0f);

      Kernels::Flat::backward_kernel(managed_stream.raw_stream(),
                                     input_accessor,
                                     input_grad_accessor.get_float_ptr(),
                                     output_grad_accessor.get_float_ptr());

      std::vector<float> backward_output_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(input_grad_accessor));

      std::vector<float> expected_output_data(
          input_accessor.shape.num_elements(), 1.0f);
      CHECK(backward_output_data == expected_output_data);
    }
  }
}
