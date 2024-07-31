#include "doctest/doctest.h"
#include "kernels/batch_norm_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test BatchNorm Kernel") {
    size_t output_n = 1, output_c = 10, output_h = 10, output_w = 10;

    ManagedFFStream managed_stream{};
    ManagedPerDeviceFFHandle managed_handle{};

    Allocator allocator = create_local_cuda_memory_allocator();

    BatchNormPerDeviceState state =
        Kernels::BatchNorm::init_kernel(managed_handle.raw_handle(),
                                        allocator,
                                        nullptr,
                                        output_n,
                                        output_c,
                                        output_h,
                                        output_w,
                                        true);

    TensorShape input_shape = make_tensor_shape_from_legion_dims(
        {output_n, output_c, output_h, output_w}, DataType::FLOAT);
    TensorShape output_shape = make_tensor_shape_from_legion_dims(
        {output_n, output_c, output_h, output_w}, DataType::FLOAT);
    TensorShape scale_shape = make_tensor_shape_from_legion_dims(
        {output_n, output_c, output_h, output_w}, DataType::FLOAT);
    TensorShape bias_shape = make_tensor_shape_from_legion_dims(
        {output_n, output_c, output_h, output_w}, DataType::FLOAT);

    GenericTensorAccessorW input_accessor =
        create_random_filled_accessor_w<DataType::FLOAT>(input_shape,
                                                         allocator);
    GenericTensorAccessorW output_accessor =
        create_random_filled_accessor_w<DataType::FLOAT>(output_shape,
                                                         allocator);
    GenericTensorAccessorW scale_accessor =
        create_filled_accessor_w<float>(scale_shape, allocator, 1.0f);

    SUBCASE("forward_kernel") {
      GenericTensorAccessorW bias_accessor =
          create_filled_accessor_w<float>(bias_shape, allocator, 0.0f);

      Kernels::BatchNorm::forward_kernel(managed_stream.raw_stream(),
                                         state,
                                         input_accessor.get_float_ptr(),
                                         output_accessor.get_float_ptr(),
                                         scale_accessor.get_float_ptr(),
                                         bias_accessor.get_float_ptr());

      std::vector<float> host_output_data =
          load_accessor_data<DataType::FLOAT>(output_accessor);
      CHECK(contains_non_zero(host_output_data));
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorW output_grad_accessor =
          create_random_filled_accessor_w<DataType::FLOAT>(output_shape,
                                                           allocator);
      GenericTensorAccessorW input_grad_accessor =
          create_random_filled_accessor_w<DataType::FLOAT>(input_shape,
                                                           allocator);
      GenericTensorAccessorW scale_grad_accessor =
          create_random_filled_accessor_w<DataType::FLOAT>(scale_shape,
                                                           allocator);
      GenericTensorAccessorW bias_grad_accessor =
          create_random_filled_accessor_w<DataType::FLOAT>(bias_shape,
                                                           allocator);

      Kernels::BatchNorm::backward_kernel(managed_stream.raw_stream(),
                                          state,
                                          input_accessor.get_float_ptr(),
                                          output_grad_accessor.get_float_ptr(),
                                          output_accessor.get_float_ptr(),
                                          input_grad_accessor.get_float_ptr(),
                                          scale_accessor.get_float_ptr(),
                                          scale_grad_accessor.get_float_ptr(),
                                          bias_grad_accessor.get_float_ptr(),
                                          input_accessor.shape.num_elements());

      std::vector<float> host_input_grad_data =
          load_accessor_data<DataType::FLOAT>(input_grad_accessor);
      std::vector<float> host_scale_grad_data =
          load_accessor_data<DataType::FLOAT>(scale_grad_accessor);
      std::vector<float> host_bias_grad_data =
          load_accessor_data<DataType::FLOAT>(bias_grad_accessor);

      CHECK(contains_non_zero(host_input_grad_data));
      CHECK(contains_non_zero(host_scale_grad_data));
      CHECK(contains_non_zero(host_bias_grad_data));
    }

    Kernels::BatchNorm::cleanup_kernel(allocator,
                                       state.inputTensor,
                                       state.biasTensor,
                                       state.outputTensor,
                                       state.actiDesc,
                                       true,
                                       state.runningMean);
  }
}
