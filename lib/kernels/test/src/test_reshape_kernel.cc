#include "doctest/doctest.h"
#include "kernels/reshape_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Reshape Forward and Backward") {
    TensorShape shape = make_float_tensor_shape_from_legion_dims({100});

    ffStream_t stream = create_ff_stream();

    Allocator allocator = get_local_memory_allocator();

    SUBCASE("forward_kernel") {
      GenericTensorAccessorR input_accessor =
          read_only_accessor_from_write_accessor(
              create_filled_accessor_w(shape, allocator, 1.0f));
      GenericTensorAccessorW output_accessor = allocator.allocate_tensor(shape);

      ReshapePerDeviceState state =
          Kernels::Reshape::init_kernel(DataType::FLOAT);

      Kernels::Reshape::forward_kernel(
          stream, state, input_accessor, output_accessor);

      std::vector<float> check_output_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(output_accessor));

      std::vector<float> expected_output_data(
          input_accessor.shape.num_elements(), 1.0f);
      CHECK(check_output_data == expected_output_data);

      SUBCASE("backward_kernel") {
        ReshapePerDeviceState state =
            Kernels::Reshape::init_kernel(DataType::FLOAT);

        Kernels::Reshape::backward_kernel(
            stream,
            state,
            output_accessor,
            read_only_accessor_from_write_accessor(output_accessor));

        std::vector<float> host_grad_input_data =
            load_data_to_host_from_device<float>(
                read_only_accessor_from_write_accessor(output_accessor));

        std::vector<float> expected_grad_input_data(
            input_accessor.shape.num_elements(), 2.0f);
        CHECK(host_grad_input_data == expected_grad_input_data);
      }
    }

    checkCUDA(cudaStreamDestroy(stream));
  }
}
