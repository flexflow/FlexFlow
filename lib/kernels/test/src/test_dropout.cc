#include "doctest/doctest.h"
#include "kernels/dropout_kernels.h"
#include "test_utils.h"
#include "utils/containers.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Dropout Kernels") {
    unsigned long long seed = 12345;
    float dropout_rate = 0.1;

    ArrayShape shape = ArrayShape{
        std::vector<size_t>{10, 10},
    };

    TensorShape input_shape =
        make_float_tensor_shape_from_legion_dims({10, 10});

    ffStream_t stream = create_ff_stream();
    PerDeviceFFHandle handle = get_per_device_ff_handle();

    Allocator allocator = get_local_memory_allocator();

    DropoutPerDeviceState state = Kernels::Dropout::init_kernel(
        handle, dropout_rate, seed, shape, allocator);

    SUBCASE("forward_kernel") {
      GenericTensorAccessorR input_data =
          read_only_accessor_from_write_accessor(
              create_random_filled_accessor_w(input_shape, allocator));
      GenericTensorAccessorW output_data =
          allocator.allocate_tensor(input_shape);
      GenericTensorAccessorW grad_input_data =
          allocator.allocate_tensor(input_shape);

      Kernels::Dropout::forward_kernel(stream,
                                       state,
                                       input_data.get_float_ptr(),
                                       output_data.get_float_ptr());

      std::vector<float> host_output_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(output_data));

      int zero_count = [&]() {
        return count(host_output_data.begin(), host_output_data.end(), 0.0f);
      }();
      float correct_zero_count = input_data.shape.num_elements() * dropout_rate;
      CHECK(zero_count == doctest::Approx(correct_zero_count).epsilon(0.5));

      SUBCASE("backward_kernel") {
        Kernels::Dropout::backward_kernel(stream,
                                          state,
                                          output_data.get_float_ptr(),
                                          grad_input_data.get_float_ptr());

        std::vector<float> host_grad_input_data =
            load_data_to_host_from_device<float>(
                read_only_accessor_from_write_accessor(grad_input_data));

        int zero_count = [&]() {
          return count(
              host_grad_input_data.begin(), host_grad_input_data.end(), 0.0f);
        }();
        float correct_zero_count =
            output_data.shape.num_elements() * dropout_rate;
        CHECK(zero_count == doctest::Approx(correct_zero_count).epsilon(0.5));
      }
    }

    Kernels::Dropout::cleanup_kernel(allocator,
                                     state.inputTensor,
                                     state.outputTensor,
                                     state.dropoutDesc,
                                     state.dropoutStates);

    cleanup_test(stream, handle);
  }
}
