#include "doctest/doctest.h"
#include "kernels/softmax_kernels.h"
#include "test_utils.h"
#include <cmath>
#include <numeric>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Softmax Kernel Operations") {
    std::size_t num_elements = 100;
    int input_n = 1, input_c = 1, input_h = 1, input_w = num_elements;

    PerDeviceFFHandle handle = get_per_device_ff_handle();
    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Allocator allocator = get_local_memory_allocator();

    TensorShape shape = make_float_tensor_shape_w_legion_dims({num_elements});

    int channels = num_elements;
    SoftmaxPerDeviceState state = Kernels::Softmax::init_kernel(
        handle, 0, input_n, channels, input_h, input_w);

    SUBCASE("forward_kernel") {
      GenericTensorAccessorW input_accessor =
          create_random_filled_accessor_w(shape, allocator);
      GenericTensorAccessorW output_accessor = allocator.allocate_tensor(shape);

      Kernels::Softmax::forward_kernel(stream,
                                       state,
                                       input_accessor.get_float_ptr(),
                                       output_accessor.get_float_ptr());

      std::vector<float> host_input_data = load_data_to_host_from_device<float>(
          read_only_accessor_from_write_accessor(input_accessor));
      std::vector<float> host_output_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(output_accessor));

      float max_input =
          *std::max_element(host_input_data.begin(), host_input_data.end());
      float sum_exp = std::accumulate(host_input_data.begin(),
                                      host_input_data.end(),
                                      0.0f,
                                      [max_input](float acc, float val) {
                                        return acc + std::exp(val - max_input);
                                      });

      for (std::size_t i = 0; i < num_elements; ++i) {
        float expected_value =
            std::exp(host_input_data[i] - max_input) / sum_exp;
        CHECK(doctest::Approx(host_output_data[i]).epsilon(0.01) ==
              expected_value);
      }

      SUBCASE("backward_kernel") {
        GenericTensorAccessorW grad_output_accessor =
            create_random_filled_accessor_w(shape, allocator);
        GenericTensorAccessorW grad_input_accessor =
            allocator.allocate_tensor(shape);

        Kernels::Softmax::backward_kernel(stream,
                                          grad_output_accessor.get_float_ptr(),
                                          grad_input_accessor.get_float_ptr(),
                                          num_elements);

        std::vector<float> check_output_data =
            load_data_to_host_from_device<float>(
                read_only_accessor_from_write_accessor(output_accessor));

        CHECK(contains_non_zero(check_output_data));
      }
    }

    cleanup_test(stream, handle);
  }
}
