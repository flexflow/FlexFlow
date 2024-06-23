#include "doctest/doctest.h"
#include "kernels/softmax_kernels.h"
#include "test_utils.h"
#include <cmath>
#include <numeric>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Softmax Kernel Operations") {
    int input_n = 1, input_c = 1, input_h = 1, input_w = 100, channels = 100;

    ManagedStream mStream = get_managed_stream();
    ManagedHandle mHandle = get_managed_handle();

    Allocator allocator = get_local_memory_allocator();

    TensorShape shape = make_float_tensor_shape_from_legion_dims({100});

    SoftmaxPerDeviceState state = Kernels::Softmax::init_kernel(
        mHandle.handle, 0, input_n, channels, input_h, input_w);

    GenericTensorAccessorW output_accessor =
        create_random_filled_accessor_w(shape, allocator);

    SUBCASE("forward_kernel") {
      GenericTensorAccessorW input_accessor =
          create_random_filled_accessor_w(shape, allocator);

      Kernels::Softmax::forward_kernel(mStream.stream,
                                       state,
                                       input_accessor.get_float_ptr(),
                                       output_accessor.get_float_ptr());

      std::vector<float> host_output_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(output_accessor));
      CHECK(contains_non_zero(host_output_data));
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorW grad_input_accessor =
          allocator.allocate_tensor(shape);

      Kernels::Softmax::backward_kernel(mStream.stream,
                                        grad_input_accessor.get_float_ptr(),
                                        output_accessor.get_float_ptr(),
                                        output_accessor.shape.num_elements());

      std::vector<float> check_output_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(output_accessor));

      CHECK(contains_non_zero(check_output_data));
    }
  }
}
