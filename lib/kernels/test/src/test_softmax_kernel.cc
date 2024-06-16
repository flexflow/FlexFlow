#include "doctest/doctest.h"
#include "kernels/softmax_kernels.h"
#include "test_utils.h"
#include <cmath>
#include <numeric>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Softmax Kernel Operations") {
    const std::size_t num_elements = 100;
    int input_n = 1, input_c = 1, input_h = 1, input_w = num_elements;

    PerDeviceFFHandle handle;
    setPerDeviceFFHandle(&handle);
    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Allocator allocator = get_local_memory_allocator();

    TensorShape shape = get_float_tensor_shape({num_elements});

    int channels = num_elements;
    SoftmaxPerDeviceState state = Kernels::Softmax::init_kernel(
        handle, 0, input_n, channels, input_h, input_w);

    SUBCASE("Test Softmax Forward") {
      GenericTensorAccessorW input_accessor =
          getRandomFilledAccessorW(shape, allocator);
      GenericTensorAccessorW output_accessor = allocator.allocate_tensor(shape);

      Kernels::Softmax::forward_kernel(stream,
                                       state,
                                       (float const *)input_accessor.ptr,
                                       (float *)output_accessor.ptr);

      std::vector<float> host_input_data =
          fill_host_data<float>(input_accessor.ptr, num_elements);
      std::vector<float> host_output_data =
          fill_host_data<float>(output_accessor.ptr, num_elements);

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

      SUBCASE("Test Softmax Backward") {
        GenericTensorAccessorW grad_output_accessor =
            getRandomFilledAccessorW(shape, allocator);
        GenericTensorAccessorW grad_input_accessor =
            allocator.allocate_tensor(shape);

        Kernels::Softmax::backward_kernel(stream,
                                          (float *)grad_output_accessor.ptr,
                                          (float *)grad_input_accessor.ptr,
                                          num_elements);

        std::vector<float> check_output_data =
            fill_host_data<float>(output_accessor.ptr, num_elements);

        REQUIRE(contains_non_zero(check_output_data));
      }
    }

    cleanup_test(stream, handle);
  }
}
