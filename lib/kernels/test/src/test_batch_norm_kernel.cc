#include "doctest/doctest.h"
#include "kernels/batch_norm_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test BatchNorm Kernel") {
    size_t output_n = 1, output_c = 10, output_h = 10, output_w = 10;
    size_t num_elements = output_n * output_c * output_h * output_w;

    PerDeviceFFHandle handle = get_per_device_ff_handle();

    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    Allocator allocator = get_local_memory_allocator();

    BatchNormPerDeviceState state = Kernels::BatchNorm::init_kernel(stream,
                                                                    handle,
                                                                    allocator,
                                                                    nullptr,
                                                                    output_n,
                                                                    output_c,
                                                                    output_h,
                                                                    output_w,
                                                                    true);

    TensorShape input_shape =
        make_float_tensor_shape_w_legion_dims({num_elements});
    TensorShape output_shape =
        make_float_tensor_shape_w_legion_dims({num_elements});
    TensorShape scale_shape = make_float_tensor_shape_w_legion_dims({output_c});
    TensorShape bias_shape = make_float_tensor_shape_w_legion_dims({output_c});

    GenericTensorAccessorW input_accessor =
        create_random_filled_accessor_w(input_shape, allocator);
    GenericTensorAccessorW output_accessor =
        allocator.allocate_tensor(output_shape);
    GenericTensorAccessorW scale_accessor =
        create_filled_accessor_w(scale_shape, allocator, 1.0f);
    GenericTensorAccessorW bias_accessor =
        create_filled_accessor_w(bias_shape, allocator, 0.0f);

    SUBCASE("forward_kernel") {
      Kernels::BatchNorm::forward_kernel(stream,
                                         state,
                                         input_accessor.get_float_ptr(),
                                         output_accessor.get_float_ptr(),
                                         scale_accessor.get_float_ptr(),
                                         bias_accessor.get_float_ptr());

      std::vector<float> host_output_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(output_accessor));
      REQUIRE(contains_non_zero(host_output_data));

      SUBCASE("backward_kernel") {
        GenericTensorAccessorW grad_output_accessor =
            create_random_filled_accessor_w(output_shape, allocator);

        Kernels::BatchNorm::backward_kernel(
            stream,
            state,
            input_accessor.get_float_ptr(),
            grad_output_accessor.get_float_ptr(),
            output_accessor.get_float_ptr(),
            input_accessor.get_float_ptr(),
            scale_accessor.get_float_ptr(),
            scale_accessor.get_float_ptr(),
            bias_accessor.get_float_ptr(),
            num_elements);

        std::vector<float> host_grad_input =
            load_data_to_host_from_device<float>(
                read_only_accessor_from_write_accessor(input_accessor));
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
