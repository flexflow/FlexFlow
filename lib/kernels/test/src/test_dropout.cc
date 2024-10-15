#include "doctest/doctest.h"
#include "kernels/dropout_kernels.h"
#include "test_utils.h"
#include "utils/containers/count.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Dropout Kernels") {
    unsigned long long seed = 12345;
    float dropout_rate = 0.1;

    ArrayShape shape = ArrayShape{
        std::vector<size_t>{10, 10},
    };

    TensorShape input_shape =
        make_tensor_shape_from_legion_dims({10, 10}, DataType::FLOAT);
    TensorShape output_shape = input_shape;

    ManagedFFStream managed_stream{};
    ManagedPerDeviceFFHandle managed_handle{};

    Allocator allocator = create_local_cuda_memory_allocator();

    DropoutPerDeviceState state = Kernels::Dropout::init_kernel(
        managed_handle.raw_handle(), dropout_rate, seed, shape, allocator);

    auto get_zero_count = [](std::vector<float> const &data) {
      return count(data, [](float x) { return x == 0.0f; });
    };

    SUBCASE("forward_kernel") {
      GenericTensorAccessorR input_accessor =
          create_random_filled_accessor_r(input_shape, allocator);
      GenericTensorAccessorW output_accessor =
          allocator.allocate_tensor(output_shape);

      Kernels::Dropout::forward_kernel(managed_stream.raw_stream(),
                                       state,
                                       input_accessor.get_float_ptr(),
                                       output_accessor.get_float_ptr());

      CHECK(contains_non_zero(output_accessor));
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorW output_grad_data =
          create_random_filled_accessor_w(output_shape, allocator);
      GenericTensorAccessorW input_grad_data =
          create_random_filled_accessor_w(input_shape, allocator);

      Kernels::Dropout::backward_kernel(managed_stream.raw_stream(),
                                        state,
                                        output_grad_data.get_float_ptr(),
                                        input_grad_data.get_float_ptr());
    }

    Kernels::Dropout::cleanup_kernel(allocator,
                                     state.inputTensor,
                                     state.outputTensor,
                                     state.dropoutDesc,
                                     state.dropoutStates);
  }
}
