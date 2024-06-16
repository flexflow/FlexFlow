#include "doctest/doctest.h"
#include "kernels/batch_norm_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test BatchNorm Kernel") {
    size_t output_n = 1, output_c = 10, output_h = 10, output_w = 10;
    size_t num_elements = output_n * output_c * output_h * output_w;

    PerDeviceFFHandle handle;
    setPerDeviceFFHandle(&handle);
    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Allocator allocator = get_local_memory_allocator();

    BatchNormPerDeviceState state = Kernels::BatchNorm::init_kernel(handle,
                                                                    allocator,
                                                                    nullptr,
                                                                    output_n,
                                                                    output_c,
                                                                    output_h,
                                                                    output_w,
                                                                    true);

    TensorShape input_shape = get_float_tensor_shape({num_elements});
    TensorShape output_shape = get_float_tensor_shape({num_elements});
    TensorShape scale_shape = get_float_tensor_shape({output_c});
    TensorShape bias_shape = get_float_tensor_shape({output_c});

    GenericTensorAccessorW input_accessor =
        getRandomFilledAccessorW(input_shape, allocator);
    GenericTensorAccessorW output_accessor =
        allocator.allocate_tensor(output_shape);
    GenericTensorAccessorW scale_accessor =
        getFilledAccessorW(scale_shape, allocator, 1.0f);
    GenericTensorAccessorW bias_accessor =
        getFilledAccessorW(bias_shape, allocator, 0.0f);

    SUBCASE("Test BatchNorm Forward") {
      Kernels::BatchNorm::forward_kernel(stream,
                                         state,
                                         (float *)input_accessor.ptr,
                                         (float *)output_accessor.ptr,
                                         (float *)scale_accessor.ptr,
                                         (float *)bias_accessor.ptr);

      std::vector<float> host_output_data =
          fill_host_data<float>(output_accessor.ptr, num_elements);
      REQUIRE(contains_non_zero(host_output_data));

      SUBCASE("Test BatchNorm Backward") {
        GenericTensorAccessorW grad_output_accessor =
            getRandomFilledAccessorW(output_shape, allocator);

        Kernels::BatchNorm::backward_kernel(stream,
                                            state,
                                            (float *)input_accessor.ptr,
                                            (float *)grad_output_accessor.ptr,
                                            (float *)output_accessor.ptr,
                                            (float *)input_accessor.ptr,
                                            (float *)scale_accessor.ptr,
                                            (float *)scale_accessor.ptr,
                                            (float *)bias_accessor.ptr,
                                            num_elements);

        std::vector<float> host_grad_input =
            fill_host_data<float>(input_accessor.ptr, num_elements);
        REQUIRE(contains_non_zero(host_grad_input));
      }
    }

    Kernels::BatchNorm::cleanup_kernel(allocator,
                                       state.inputTensor,
                                       state.biasTensor,
                                       state.outputTensor,
                                       state.actiDesc,
                                       true,
                                       nullptr);
    cleanup_test(stream, handle);
  }
}
