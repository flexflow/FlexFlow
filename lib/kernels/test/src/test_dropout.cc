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

    ManagedStream mStream = get_managed_stream();
    ManagedHandle mHandle = get_managed_handle();

    Allocator allocator = get_local_memory_allocator();

    DropoutPerDeviceState state = Kernels::Dropout::init_kernel(
        mHandle.handle, dropout_rate, seed, shape, allocator);

    auto get_zero_count = [](std::vector<float> const &data) {
      return count(data, [](float x) { return x == 0.0f; });
    };

    GenericTensorAccessorW output_data =
        create_random_filled_accessor_w(input_shape, allocator);
    GenericTensorAccessorW grad_input_data =
        create_random_filled_accessor_w(input_shape, allocator);

    SUBCASE("forward_kernel") {
      GenericTensorAccessorR input_data =
          read_only_accessor_from_write_accessor(
              create_random_filled_accessor_w(input_shape, allocator));

      Kernels::Dropout::forward_kernel(mStream.stream,
                                       state,
                                       input_data.get_float_ptr(),
                                       output_data.get_float_ptr());

      std::vector<float> host_output_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(output_data));

      CHECK(contains_non_zero(host_output_data));
    }

    SUBCASE("backward_kernel") {
      Kernels::Dropout::backward_kernel(mStream.stream,
                                        state,
                                        output_data.get_float_ptr(),
                                        grad_input_data.get_float_ptr());
    }

    Kernels::Dropout::cleanup_kernel(allocator,
                                     state.inputTensor,
                                     state.outputTensor,
                                     state.dropoutDesc,
                                     state.dropoutStates);
  }
}
