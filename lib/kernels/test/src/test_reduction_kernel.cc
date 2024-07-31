#include "doctest/doctest.h"
#include "kernels/reduction_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Reduction Forward and Backward Kernel") {
    std::size_t num_replicas = 5;

    TensorShape input_shape = make_tensor_shape_from_legion_dims(
        {10, 10, 10, 10, 10}, DataType::FLOAT);

    ManagedPerDeviceFFHandle managed_handle{};
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    SUBCASE("forward_kernel") {
      TensorShape output_shape =
          make_tensor_shape_from_legion_dims({10}, DataType::FLOAT);

      GenericTensorAccessorR input_accessor =
          create_random_filled_accessor_r<DataType::FLOAT>(input_shape,
                                                           allocator);
      GenericTensorAccessorW output_accessor =
          allocator.allocate_tensor(output_shape);

      Kernels::Reduction::forward_kernel(managed_stream.raw_stream(),
                                         input_accessor,
                                         output_accessor,
                                         num_replicas);

      std::vector<float> host_output_data =
          load_accessor_data<DataType::FLOAT>(output_accessor);
      CHECK(contains_non_zero(host_output_data));
    }

    SUBCASE("backward_kernel") {
      TensorShape output_shape = input_shape;

      GenericTensorAccessorR output_grad_accessor =
          read_only_accessor_from_write_accessor(
              create_filled_accessor_w<float>(output_shape, allocator, 1.0f));
      GenericTensorAccessorW input_grad_accessor =
          allocator.allocate_tensor(input_shape);

      Kernels::Reduction::backward_kernel(managed_stream.raw_stream(),
                                          input_grad_accessor,
                                          output_grad_accessor);

      std::vector<float> expected_grad_input_data(
          input_grad_accessor.shape.num_elements(), 1.0f);
      std::vector<float> host_grad_data =
          load_accessor_data<DataType::FLOAT>(input_grad_accessor);
      CHECK(host_grad_data == expected_grad_input_data);
    }
  }
}
