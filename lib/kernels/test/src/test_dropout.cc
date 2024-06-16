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

    TensorShape input_shape = get_float_tensor_shape({num_elements});

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    PerDeviceFFHandle handle;
    setPerDeviceFFHandle(&handle);

    Allocator allocator = get_local_memory_allocator();

    DropoutPerDeviceState state = Kernels::Dropout::init_kernel(
        handle, dropout_rate, seed, shape, allocator);

    GenericTensorAccessorR input_data =
        makeReadOnlyAccessor(getRandomFilledAccessorW(input_shape, allocator));
    GenericTensorAccessorW output_data = allocator.allocate_tensor(input_shape);
    GenericTensorAccessorW grad_input_data =
        allocator.allocate_tensor(input_shape);

    SUBCASE("Test Dropout Forward") {
      Kernels::Dropout::forward_kernel(stream,
                                       state,
                                       (float const *)input_data.ptr,
                                       (float *)output_data.ptr);

      std::vector<float> host_output_data =
          fill_host_data<float>(output_data.ptr, num_elements);

      int zero_count = 0;
      for (float value : host_output_data) {
        if (value == 0.0f) {
          zero_count++;
        }
      }
      CHECK(zero_count ==
            doctest::Approx(num_elements * dropout_rate).epsilon(0.5));

      SUBCASE("Test Dropout Backward") {
        Kernels::Dropout::backward_kernel(stream,
                                          state,
                                          (float const *)output_data.ptr,
                                          (float *)grad_input_data.ptr);

        std::vector<float> host_grad_input_data =
            fill_host_data<float>(grad_input_data.ptr, num_elements);

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
