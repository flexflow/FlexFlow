#include "doctest/doctest.h"
#include "kernels/partition_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Partition Forward and Backward") {
    ManagedPerDeviceFFHandle managed_handle{};
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    RepartitionPerDeviceState state = Kernels::Repartition::init_kernel(
        managed_handle.raw_handle(), DataType::FLOAT);

    TensorShape input_shape =
        make_tensor_shape_from_legion_dims({10, 10}, DataType::FLOAT);
    TensorShape output_shape = input_shape;

    SUBCASE("forward_kernel") {
      GenericTensorAccessorR input_accessor =
          read_only_accessor_from_write_accessor(
              create_filled_accessor_w<float>(input_shape, allocator, 1.0f));
      GenericTensorAccessorW output_accessor =
          allocator.allocate_tensor(output_shape);

      Kernels::Repartition::forward_kernel(
          managed_stream.raw_stream(), state, input_accessor, output_accessor);

      std::vector<float> check_output_data =
          load_accessor_data<DataType::FLOAT>(output_accessor);

      std::vector<float> expected_output_data(
          input_accessor.shape.num_elements(), 1.0f);
      CHECK(check_output_data == expected_output_data);
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorR output_grad_accessor =
          read_only_accessor_from_write_accessor(
              create_filled_accessor_w<float>(output_shape, allocator, 1.0f));
      GenericTensorAccessorW input_grad_accessor =
          create_filled_accessor_w<float>(input_shape, allocator, 2.0f);

      Kernels::Repartition::backward_kernel(managed_stream.raw_stream(),
                                            state,
                                            input_grad_accessor,
                                            output_grad_accessor);

      std::vector<float> host_grad_input_data =
          load_accessor_data<DataType::FLOAT>(input_grad_accessor);

      std::vector<float> expected_grad_input_data(
          input_grad_accessor.shape.num_elements(), 3.0f);
      CHECK(host_grad_input_data == expected_grad_input_data);
    }
  }
}
