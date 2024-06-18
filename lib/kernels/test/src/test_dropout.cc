#include "doctest/doctest.h"
#include "kernels/dropout_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Dropout Kernels") {
    unsigned long long seed = 12345;
    float dropout_rate = 0.1;
    std::size_t num_elements = 100;

    ArrayShape shape = ArrayShape{
        std::vector<size_t>{100, 100},
    };

    TensorShape input_shape =
        make_float_tensor_shape_w_legion_dims({num_elements});

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    PerDeviceFFHandle handle = get_per_device_ff_handle();

    Allocator allocator = get_local_memory_allocator();

    DropoutPerDeviceState state = Kernels::Dropout::init_kernel(
        handle, dropout_rate, seed, shape, allocator);

    GenericTensorAccessorR input_data = read_only_accessor_from_write_accessor(
        create_random_filled_accessor_w(input_shape, allocator));
    GenericTensorAccessorW output_data = allocator.allocate_tensor(input_shape);
    GenericTensorAccessorW grad_input_data =
        allocator.allocate_tensor(input_shape);

    SUBCASE("forward_kernel") {
      Kernels::Dropout::forward_kernel(stream,
                                       state,
                                       input_data.get_float_ptr(),
                                       output_data.get_float_ptr());

      std::vector<float> host_output_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(output_data));

      int zero_count = [&]() {
        return std::count_if(host_output_data.begin(),
                             host_output_data.end(),
                             [](float value) { return value == 0.0f; });
      }();

      float correct_zero_count = num_elements * dropout_rate;
      CHECK(zero_count == doctest::Approx(correct_zero_count).epsilon(0.5));

      SUBCASE("backward_kernel") {
        Kernels::Dropout::backward_kernel(stream,
                                          state,
                                          output_data.get_float_ptr(),
                                          grad_input_data.get_float_ptr());

        std::vector<float> host_grad_input_data =
            load_data_to_host_from_device<float>(
                read_only_accessor_from_write_accessor(grad_input_data));

        int zero_count = 0;
        for (float value : host_grad_input_data) {
          if (value == 0.0f) {
            zero_count++;
          }
        }
        CHECK(zero_count ==
              doctest::Approx(num_elements * dropout_rate).epsilon(0.5));
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
